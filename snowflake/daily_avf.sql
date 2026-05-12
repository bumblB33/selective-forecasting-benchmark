-- TEMPLATE: Daily actual-vs-forecast extract
--
-- Before running, replace all placeholder tokens:
--   <your_db>                   Your Snowflake database name
--   <your_forecasting_schema>   Schema containing forecast sessions and interval forecasts
--   <your_sales_schema>         Schema containing daily and ticket-level sales actuals
--   <your_precomputed_data_source>  Table/view exposing IS_OUTLIER flags per location-day
--
-- Expected output columns (loaded by daily/data_loader.py):
--   ENTITY_UID              VARCHAR   -- CUSTOMER_ID || '_' || UPPER(SITE_ID)
--   CUSTOMER_ID             INTEGER
--   SITE_ID                 VARCHAR
--   DAY_START_TIME          TIMESTAMP
--   PROJECTION              FLOAT     -- Net sales projection (baseline forecast)
--   USER_ADJUSTMENT         FLOAT
--   PUBLISHED_USER_ADJUSTMENT FLOAT
--   ACTUAL_VALUE            FLOAT     -- Net sales actuals (use with --metric net_sales)
--   GROSS_SALES_ACTUAL      FLOAT     -- Gross sales actuals (use with --metric gross_sales)
--   IS_GENERATED_BY_MANAGER BOOLEAN
--   IS_SYSTEMATIC_GENERATED BOOLEAN
--   IS_OUTLIER              BOOLEAN   -- TRUE when the production system excluded this day from training
--   OUTLIER_EXCLUSION_REASON INTEGER  -- NULL when IS_OUTLIER is FALSE
--
-- The query samples up to target_location_count locations with sufficient history.
-- Adjust params CTE values to match your data volume and evaluation window.

WITH params AS (
    SELECT
        1     AS net_sales_metric_id,        -- Metric 1 = NetSales in forecasting schema
        90    AS min_days_with_actuals,
        300   AS target_location_count,
        CURRENT_DATE() - INTERVAL '3 YEARS'  AS actual_vs_forecast_start,
        CURRENT_DATE()                        AS actual_vs_forecast_end  -- Exclusive upper bound
),

-- Latest daily forecast session per location-day.
-- ID is the FK referenced by DAILY_INTERVAL_FORECASTS.DAILY_FORECAST_ID.
actual_vs_forecast_sessions AS (
    SELECT
        id                            AS session_id,
        tenant_id,
        UPPER(location_id)            AS location_id,
        day_start_time,
        day_start_time::DATE          AS day_date,
        is_generated_by_manager,
        is_systematic_generated
    FROM <your_db>.<your_forecasting_schema>.daily_forecast_sessions
    -- NOTE: IS_OUTLIER and OUTLIER_EXCLUSION_REASON are NOT available on daily_forecast_sessions.
    -- They live on the precompute data document (DbPrecomputeDataPointBy15Minutes in MongoDB).
    -- The precompute_data CTE below joins them in via the precompute Snowflake connector table.
    WHERE
        metric_id      = (SELECT net_sales_metric_id FROM params)
        AND day_start_time >= (SELECT actual_vs_forecast_start FROM params)
        AND day_start_time <  (SELECT actual_vs_forecast_end   FROM params)
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY tenant_id, location_id, metric_id, day_start_time::DATE
        ORDER BY created_on DESC, event_timestamp DESC  -- Latest session wins
    ) = 1
),

-- Daily projection totals joined from the normalized interval forecast table.
-- Replaces LATERAL FLATTEN(interval_forecasts::ARRAY) + BSON TRY_CAST parsing.
interval_agg AS (
    SELECT
        s.tenant_id,
        s.location_id,
        s.day_date,
        s.day_start_time,
        s.is_generated_by_manager,
        s.is_systematic_generated,
        SUM(f.projection)                   AS projection,
        SUM(f.user_adjustment)              AS user_adjustment,
        SUM(f.published_user_adjustment)    AS published_user_adjustment
    FROM actual_vs_forecast_sessions AS s
    JOIN <your_db>.<your_forecasting_schema>.daily_interval_forecasts AS f
        ON  f.daily_forecast_id = s.session_id
        AND f.tenant_id         = s.tenant_id
    GROUP BY
        s.tenant_id, s.location_id, s.day_date, s.day_start_time,
        s.is_generated_by_manager, s.is_systematic_generated
),

-- Daily actuals from the operational sales summary table.
-- Replaces LATERAL FLATTEN(comparisons::ARRAY) + BSON TRY_CAST + ComparisonType=1 filter.
-- Deduplicates to the latest version of each location-day record.
-- Both net_sales and gross_sales are extracted to support multi-metric benchmarking.
actual_vs_forecast_actuals AS (
    SELECT
        tenant_id,
        UPPER(location_id) AS location_id,
        date::DATE         AS day_date,
        net_sales          AS actual_value,
        gross_sales        AS gross_sales_actual
    FROM <your_db>.<your_sales_schema>.daily_sales_summary
    WHERE
        date::DATE >= (SELECT actual_vs_forecast_start FROM params)
        AND date::DATE <  (SELECT actual_vs_forecast_end   FROM params)
        AND is_deleted    = FALSE
        AND is_future_dss = FALSE
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY tenant_id, location_id, date::DATE
        ORDER BY event_timestamp DESC
    ) = 1
),

-- Locations with sufficient history for model training.
-- Filters to days with valid actuals AND positive projections (excludes closed days,
-- zero-forecast periods). projection > 0 ensures we evaluate only intervals Baseline
-- actively forecasted.
eligible_locations AS (
    SELECT ia.tenant_id, ia.location_id
    FROM interval_agg AS ia
    JOIN actual_vs_forecast_actuals AS a
        ON  ia.tenant_id   = a.tenant_id
        AND ia.location_id = a.location_id
        AND ia.day_date    = a.day_date
    WHERE a.actual_value IS NOT NULL
      AND ia.projection > 0
    GROUP BY ia.tenant_id, ia.location_id
    HAVING COUNT(DISTINCT day_date) >= (SELECT min_days_with_actuals FROM params)
),

-- Random sample of eligible locations for benchmark evaluation.
-- Uses RANDOM() for unbiased sampling (HASH can introduce systematic tenant-level bias).
sampled_locations AS (
    SELECT tenant_id, location_id
    FROM eligible_locations
    QUALIFY ROW_NUMBER() OVER (ORDER BY RANDOM()) <= (SELECT target_location_count FROM params)
),

-- Day-level outlier flags from the precompute collection.
-- ProductionForecastingService writes IS_OUTLIER=TRUE on days excluded from training by
-- ForecastingRules (ByForecastingRule) or ProjectionDates review (ByProjectionDates).
-- These days are dropped entirely from model training in production — they must be
-- dropped here too so the benchmark trains on the same data.
--
-- TODO: Replace <your_db>.<your_forecasting_schema>.<your_precomputed_data_source> with the verified
--       Snowflake connector table name for DbPrecomputeDataPointBy15Minutes.
--       The join key is (tenant_id, location_id, day_date).
precompute_outlier_flags AS (
    SELECT
        tenant_id,
        UPPER(location_id)            AS location_id,
        day_start_time::DATE          AS day_date,
        is_outlier,
        outlier_exclusion_reason
    FROM <your_db>.<your_forecasting_schema>.<your_precomputed_data_source>
    WHERE
        day_start_time >= (SELECT actual_vs_forecast_start FROM params)
        AND day_start_time <  (SELECT actual_vs_forecast_end   FROM params)
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY tenant_id, location_id, day_start_time::DATE
        ORDER BY updated_on DESC  -- Latest precompute document wins
    ) = 1
)

-- Final output includes ALL days for sampled locations (both eligible and ineligible days)
-- to provide complete historical context. Python pipeline handles further filtering.
-- IS_OUTLIER / OUTLIER_EXCLUSION_REASON are LEFT-joined: days with no precompute record
-- default to IS_OUTLIER=FALSE so the Python loader can treat NULL as non-outlier without
-- silently dropping rows that simply predate the precompute collection window.
SELECT
    ia.tenant_id::VARCHAR || '_' || UPPER(ia.location_id)  AS ENTITY_UID,
    ia.tenant_id                                           AS CUSTOMER_ID,
    ia.location_id                                         AS SITE_ID,
    ia.day_start_time                                      AS DAY_START_TIME,
    -- COALESCE defensive: projection/adjustments are SUM() aggregates that should never
    -- be NULL for rows in interval_agg, but guarding against edge cases where a session
    -- exists with zero matching interval forecasts.
    COALESCE(ia.projection, 0.0)                           AS PROJECTION,
    COALESCE(ia.user_adjustment, 0.0)                      AS USER_ADJUSTMENT,
    COALESCE(ia.published_user_adjustment, 0.0)            AS PUBLISHED_USER_ADJUSTMENT,
    a.actual_value                                         AS ACTUAL_VALUE,
    a.gross_sales_actual                                   AS GROSS_SALES_ACTUAL,
    ia.is_generated_by_manager                             AS IS_GENERATED_BY_MANAGER,
    ia.is_systematic_generated                             AS IS_SYSTEMATIC_GENERATED,
    -- COALESCE to FALSE: a missing precompute record means the day was never flagged as
    -- an outlier by ProductionForecastingService, so it is safe to include in training.
    COALESCE(pof.is_outlier, FALSE)                        AS IS_OUTLIER,
    pof.outlier_exclusion_reason                           AS OUTLIER_EXCLUSION_REASON
FROM interval_agg AS ia
JOIN actual_vs_forecast_actuals AS a
    ON  ia.tenant_id   = a.tenant_id
    AND ia.location_id = a.location_id
    AND ia.day_date    = a.day_date
JOIN sampled_locations AS sl
    ON  ia.tenant_id   = sl.tenant_id
    AND ia.location_id = sl.location_id
LEFT JOIN precompute_outlier_flags AS pof
    ON  ia.tenant_id   = pof.tenant_id
    AND ia.location_id = pof.location_id
    AND ia.day_date    = pof.day_date
ORDER BY ia.tenant_id, ia.location_id, ia.day_start_time;
