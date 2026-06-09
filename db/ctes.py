"""Reusable SQL Common Table Expressions (CTEs) for queries and visualizations.

Keep shared CTEs here so multiple scripts import the same string and stay
consistent when the pricing schema changes.
"""

LATEST_PRICING_CTE = """
latest_pricing AS (
    SELECT
        model_id,
        input_cost_per_1m,
        output_cost_per_1m,
        (input_cost_per_1m + output_cost_per_1m) / 2.0 AS avg_cost_per_1m,
        effective_date,
        ROW_NUMBER() OVER (
            PARTITION BY model_id
            ORDER BY effective_date DESC
        ) AS rn
    FROM pricing
)
"""
