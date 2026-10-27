-- TEMPLATE: 15-minute interval actual-vs-forecast extract
--
-- Before running, replace all placeholder tokens:
--   <your_db>                   Your Snowflake database name
--   <your_forecasting_schema>   Schema containing forecast sessions and interval forecasts
--   <your_sales_schema>         Schema containing ticket-level sales actuals
--   <your_precomputed_data_source>  Table/view exposing IS_OUTLIER flags per location-day
--
-- Expected output columns (loaded by interval/data_loader.py):
--   ENTITY_UID                VARCHAR  -- CUSTOMER_ID || '_' || UPPER(SITE_ID)
--   CUSTOMER_ID               INTEGER
--   SITE_ID                   VARCHAR  -- uppercased
--   INTERVAL_ID_TICKS         BIGINT   -- downstream system tick key for this interval
--   INTERVAL_START_TS         TIMESTAMP
--   NET_SALES                 FLOAT    -- actual interval net sales (NULL if no sales record)
--   GROSS_SALES               FLOAT    -- actual interval gross sales
--   BASELINE_PROJECTION       FLOAT    -- production system per-interval projection
--   USER_ADJUSTMENT           FLOAT
--   PUBLISHED_USER_ADJUSTMENT FLOAT
--   IS_CLOSED                 BOOLEAN
--   BUSINESS_START_HOUR_UTC   INTEGER  -- per-location business-day start hour in UTC
--   DAILY_NET_SALES           FLOAT    -- daily total for this interval's business day
--   IS_OUTLIER                BOOLEAN
--   OUTLIER_EXCLUSION_REASON  INTEGER  -- NULL when IS_OUTLIER is FALSE
--
-- INTERVAL_ID_TICKS formula (adapt epoch to your system):
--   UNIX_SECONDS × 10,000,000 + <ticks_at_unix_epoch_for_your_system>
--
-- The query samples up to target_location_count locations with sufficient history.
-- Adjust params CTE values to match your data volume and evaluation window.

WITH params AS (
    SELECT
        1     AS net_sales_metric_id,        -- Metric 1 = NetSales in forecasting schema
        26    AS train_weeks,
        1     AS test_weeks,
        28    AS min_days_with_actuals,
        0.5   AS min_coverage_fraction,   -- fraction of expected slots required
        300   AS target_location_count,
        '2024-01-01'::DATE AS actual_vs_forecast_start,
        CURRENT_DATE()     AS actual_vs_forecast_end  -- Exclusive upper bound
),

actual_vs_forecast_sessions AS (
    SELECT
        id                            AS session_id,
        tenant_id,
        UPPER(location_id)            AS location_id,
        day_start_time::DATE          AS day_date,
        interval_forecasts::ARRAY     AS interval_forecasts
    FROM <your_db>.<your_forecasting_schema>.daily_forecast_sessions
    WHERE
        metric_id      = (SELECT net_sales_metric_id FROM params)
        AND day_start_time >= (SELECT actual_vs_forecast_start FROM params)
        AND day_start_time <  (SELECT actual_vs_forecast_end   FROM params)
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY tenant_id, location_id, metric_id, day_start_time::DATE
        ORDER BY created_on DESC, event_timestamp DESC  -- Latest session wins
    ) = 1
),

-- Mirrors daily_avf.sql: uses normalized daily_interval_forecasts instead of
-- BSON LATERAL FLATTEN so eligibility logic is consistent across both pipelines
-- and survives any future BSON schema changes.
actual_vs_forecast_projections AS (
    SELECT
        s.tenant_id,
        s.location_id,
        s.day_date,
        SUM(f.projection) AS total_projection
    FROM actual_vs_forecast_sessions AS s
    JOIN <your_db>.<your_forecasting_schema>.daily_interval_forecasts AS f
        ON  f.daily_forecast_id = s.session_id
        AND f.tenant_id         = s.tenant_id
    GROUP BY s.tenant_id, s.location_id, s.day_date
),

-- Mirrors daily_avf.sql: uses daily_sales_summary as the authoritative actuals
-- source. This also fixes real_actuals_tail (below) so the training window
-- anchor comes from the same table as the actual training data.
actual_vs_forecast_actuals AS (
    SELECT
        tenant_id,
        UPPER(location_id) AS location_id,
        date::DATE         AS day_date,
        net_sales          AS actual_value
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

eligible_locations AS (
    SELECT p.tenant_id, p.location_id
    FROM actual_vs_forecast_projections AS p
    JOIN actual_vs_forecast_actuals AS a
        ON  p.tenant_id   = a.tenant_id
        AND p.location_id = a.location_id
        AND p.day_date    = a.day_date
    WHERE a.actual_value IS NOT NULL
      AND p.total_projection > 0
    GROUP BY p.tenant_id, p.location_id
    HAVING COUNT(DISTINCT p.day_date) >= (SELECT min_days_with_actuals FROM params)
),

real_actuals_tail AS (
    SELECT COALESCE(MAX(day_date), '1900-01-01'::DATE) AS tail_date
    FROM actual_vs_forecast_actuals
    WHERE actual_value > 0
),

interval_bounds AS (
    SELECT
        DATEADD(WEEK,
                -((SELECT train_weeks FROM params) + (SELECT test_weeks FROM params)),
                DATEADD(DAY, 1, tail_date)) AS data_start,
        DATEADD(DAY, 1, tail_date)          AS data_end    -- exclusive
    FROM real_actuals_tail
),

dedup_intervals AS (
    SELECT
        i.tenant_id,
        UPPER(i.location_id)          AS location_id,
        i.interval_start_timestamp,
        i.net_sales,
        i.gross_sales
    FROM <your_db>.<your_sales_schema>.sales_ticket_with_interval AS i
    JOIN eligible_locations AS el
        ON  i.tenant_id          = el.tenant_id
        AND UPPER(i.location_id) = UPPER(el.location_id)
    JOIN interval_bounds AS b
        ON  i.interval_date >= b.data_start
        AND i.interval_date <  b.data_end
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY i.tenant_id, i.id, i.interval_id,
                     i.interval_start_timestamp, i.location_id
        ORDER BY i.net_sales DESC, i.gross_sales DESC
    ) = 1
),

actual_intervals AS (
    SELECT
        tenant_id,
        location_id,
        interval_start_timestamp,
        SUM(net_sales)                                                           AS net_sales,
        SUM(gross_sales)                                                         AS gross_sales,
        COUNT(*) OVER (PARTITION BY tenant_id, location_id)                      AS location_slot_count
    FROM dedup_intervals
    GROUP BY tenant_id, location_id, interval_start_timestamp
),

min_slots AS (
    SELECT FLOOR(train_weeks * 7 * 96 * min_coverage_fraction)::INT AS minimum_required_slots
    FROM params
),

-- Filters locations by coverage guard: must have at least min_coverage_fraction of expected
-- training slots (e.g., 50% of train_weeks * 7 * 96 intervals), then randomly samples
-- target_location_count. Uses RANDOM() for unbiased sampling (HASH can introduce
-- systematic tenant-level bias).
sampled_locations AS (
    SELECT DISTINCT tenant_id, location_id
    FROM actual_intervals
    WHERE location_slot_count >= (SELECT minimum_required_slots FROM min_slots)
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY 1
        ORDER BY RANDOM()  -- Unbiased random sampling
    ) <= (SELECT target_location_count FROM params)
),

-- Per-location business-day start hour in UTC, derived from the daily_forecast_sessions
-- day_start_time timestamp.  Baseline stores day_start_time as the UTC moment when the
-- business day begins (e.g. 2 AM EST = 07:00 UTC for a UTC-5 location with VSD=2).
-- PERCENTILE_DISC(0.5) uses the median across all sessions, which handles DST-induced
-- ±1 h variation without being thrown off by a single anomalous session.
location_vsd AS (
    SELECT
        dfs.tenant_id,
        UPPER(dfs.location_id)                                                     AS location_id,
        PERCENTILE_DISC(0.5) WITHIN GROUP (
            ORDER BY EXTRACT(HOUR FROM dfs.day_start_time)::INT
        )                                                                          AS business_start_hour_utc
    FROM <your_db>.<your_forecasting_schema>.daily_forecast_sessions AS dfs
    JOIN sampled_locations AS sl
        ON  dfs.tenant_id          = sl.tenant_id
        AND UPPER(dfs.location_id) = sl.location_id
    WHERE dfs.metric_id  = (SELECT net_sales_metric_id FROM params)
      AND dfs.day_start_time >= (SELECT actual_vs_forecast_start FROM params)
      AND dfs.day_start_time <  (SELECT actual_vs_forecast_end   FROM params)
    GROUP BY dfs.tenant_id, UPPER(dfs.location_id)
),

-- Day-level outlier flags from the precompute collection.
-- ProductionForecastingService writes IS_OUTLIER=TRUE on days excluded from training by
-- ForecastingRules (ByForecastingRule) or ProjectionDates review (ByProjectionDates).
-- Joined on business_day_date (not calendar date) so VSD boundaries are respected —
-- an interval at 01:00 UTC for a location with VSD=07:00 UTC belongs to the prior
-- business day, and the outlier flag for that business day must match.
--
-- TODO: Replace <your_db>.<your_forecasting_schema>.<your_precomputed_data_source> with the verified
--       Snowflake connector table name for DbPrecomputeDataPointBy15Minutes.
--       The join key is (tenant_id, location_id, business_day_date).
precompute_outlier_flags AS (
    SELECT
        tenant_id,
        UPPER(location_id)            AS location_id,
        day_start_time::DATE          AS day_date,
        is_outlier,
        outlier_exclusion_reason
    FROM <your_db>.<your_forecasting_schema>.<your_precomputed_data_source>
    JOIN sampled_locations AS sl
        ON  tenant_id          = sl.tenant_id
        AND UPPER(location_id) = sl.location_id
    WHERE
        day_start_time >= (SELECT actual_vs_forecast_start FROM params)
        AND day_start_time <  (SELECT actual_vs_forecast_end   FROM params)
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY tenant_id, location_id, day_start_time::DATE
        ORDER BY updated_on DESC  -- Latest precompute document wins
    ) = 1
),

-- Each element in interval_forecasts carries IntervalId (.NET ticks) that
-- matches the ticks we compute from interval_start_timestamp, enabling a
-- direct join without timestamp parsing.

baseline_interval_projections AS (
    SELECT
        s.tenant_id,
        UPPER(s.location_id)                                                     AS location_id,
        f.value:IntervalId."$numberLong"::BIGINT                                 AS interval_id_ticks,
        TRY_CAST(f.value:Projection."$numberDouble"::STRING
            AS NUMBER(38, 4))                                                    AS baseline_projection,
        TRY_CAST(f.value:UserAdjustment."$numberDouble"::STRING
            AS NUMBER(38, 4))                                                    AS user_adjustment,
        TRY_CAST(f.value:PublishedUserAdjustment."$numberDouble"::STRING
            AS NUMBER(38, 4))                                                    AS published_user_adjustment,
        f.value:IsClosed::BOOLEAN                                                AS is_closed
    FROM actual_vs_forecast_sessions AS s
    JOIN sampled_locations AS sl
        ON  s.tenant_id          = sl.tenant_id
        AND UPPER(s.location_id) = sl.location_id,
    LATERAL FLATTEN(input => s.interval_forecasts) AS f
    WHERE f.value:IntervalId IS NOT NULL
),

interval_grid AS (
    SELECT
        a.tenant_id,
        a.location_id,
        (EXTRACT(EPOCH FROM a.interval_start_timestamp)::BIGINT * 10000000
            + 621355968000000000)::BIGINT                                        AS interval_id_ticks,
        a.interval_start_timestamp,
        a.net_sales,
        a.gross_sales,
        r.baseline_projection,
        r.user_adjustment,
        r.published_user_adjustment,
        r.is_closed,
        COALESCE(v.business_start_hour_utc, 0)                                   AS business_start_hour_utc,
        -- Business day date: if interval falls before the business-day start hour in UTC
        -- it belongs to the previous calendar day's business day (VSD roll-back).
        CASE
            WHEN EXTRACT(HOUR FROM a.interval_start_timestamp)::INT
                     < COALESCE(v.business_start_hour_utc, 0)
            THEN DATEADD(DAY, -1, a.interval_start_timestamp::DATE)
            ELSE a.interval_start_timestamp::DATE
        END                                                                      AS business_day_date
    FROM actual_intervals AS a
    JOIN sampled_locations AS sl
        ON  a.tenant_id   = sl.tenant_id
        AND a.location_id = sl.location_id
    LEFT JOIN baseline_interval_projections AS r
        ON  a.tenant_id   = r.tenant_id
        AND a.location_id = r.location_id
        AND (EXTRACT(EPOCH FROM a.interval_start_timestamp)::BIGINT * 10000000
                + 621355968000000000) = r.interval_id_ticks
    LEFT JOIN location_vsd AS v
        ON  a.tenant_id   = v.tenant_id
        AND a.location_id = v.location_id
)

-- IS_OUTLIER / OUTLIER_EXCLUSION_REASON are LEFT-joined on business_day_date so VSD-aware
-- day boundaries match exactly what ProductionForecastingService uses during training exclusion.
-- Days with no precompute record COALESCE to FALSE — they were never flagged as outliers.
SELECT
    ig.tenant_id::VARCHAR || '_' || ig.location_id  AS ENTITY_UID,
    ig.tenant_id                                     AS CUSTOMER_ID,
    ig.location_id                                   AS SITE_ID,
    ig.interval_id_ticks                             AS INTERVAL_ID_TICKS,
    ig.interval_start_timestamp                      AS INTERVAL_START_TS,
    ig.net_sales                                     AS NET_SALES,
    ig.gross_sales                                   AS GROSS_SALES,
    ig.baseline_projection                               AS BASELINE_PROJECTION,
    ig.user_adjustment                               AS USER_ADJUSTMENT,
    ig.published_user_adjustment                     AS PUBLISHED_USER_ADJUSTMENT,
    ig.is_closed                                     AS IS_CLOSED,
    ig.business_start_hour_utc                       AS BUSINESS_START_HOUR_UTC,
    act.actual_value                                 AS DAILY_NET_SALES,
    -- COALESCE to FALSE: a missing precompute record means the day was never flagged as
    -- an outlier by ProductionForecastingService, so it is safe to include in training.
    COALESCE(pof.is_outlier, FALSE)                  AS IS_OUTLIER,
    pof.outlier_exclusion_reason                     AS OUTLIER_EXCLUSION_REASON
FROM interval_grid AS ig
LEFT JOIN actual_vs_forecast_actuals AS act
    ON  ig.tenant_id        = act.tenant_id
    AND ig.location_id      = act.location_id
    AND ig.business_day_date = act.day_date
LEFT JOIN precompute_outlier_flags AS pof
    ON  ig.tenant_id         = pof.tenant_id
    AND ig.location_id       = pof.location_id
    AND ig.business_day_date = pof.day_date
ORDER BY ig.tenant_id, ig.location_id, ig.interval_start_timestamp;
