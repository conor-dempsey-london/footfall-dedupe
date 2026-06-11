# %%
import pandas as pd
import time
from sqlalchemy import select, select, func, cast, case, literal, literal_column, text
import sqlparse

from datetime import date

import hs_models.utils as util 

from hs_models.constants import (
    FOOTFALL_COUNTS_TABLE,
    HEX_GEOM_TABLE,
    HS_TABLE,
    TC_TABLE,
    BID_TABLE,
)

start_date = date.fromisoformat('2024-01-01')
end_date = date.fromisoformat('2025-08-16')

metadata, engine = util.get_db_metadata()

area_t = metadata.tables[HS_TABLE]
hex_geom_t = metadata.tables[HEX_GEOM_TABLE]
measure_table = metadata.tables[FOOTFALL_COUNTS_TABLE]

# %%

single_id_subquery = select(area_t.c.highstreet_id).limit(1).cte("target_area_id")

# Overwrite the variable with a filtered CTE. 
# Because this object still uses '.c', all code below remains identical.
# area_t = (
#     select(area_t)
#     .join(single_id_subquery, area_t.c.highstreet_id == single_id_subquery.c.highstreet_id)
#     .cte("filtered_single_area")
# )

measure_select = select(measure_table)

if start_date is not None:
    measure_select = measure_select.where(measure_table.c.count_date >= start_date)
if end_date is not None:
    # Extend the boundary by +1 day so tomorrow's '00-03' row is included
    measure_select = measure_select.where(measure_table.c.count_date <= func.cast(end_date + literal(1), measure_table.c.count_date.type))

# Overwrite the variable with the CTE. 
measure_t = measure_select.cte("filtered_measurements")

# Build the 'spatial_weights' CTE
# We calculate the overlap fraction for each intersecting pair 
# i.e. the fraction of the hex area that is within the area boundary
overlap_fraction = case(
    (func.ST_Within(hex_geom_t.c.geom, area_t.c.geom), 1.0),
    else_=func.ST_Area(func.ST_Intersection(hex_geom_t.c.geom, area_t.c.geom)) 
        / func.nullif(func.ST_Area(hex_geom_t.c.geom), 0)
)

# %%
TEST_AREA_IDS= [619, 1, 2, 3, 4]

# spatial join areas to hexes and then return an area-hex lookup
# including the fraction of each hex that is within each area
overlap_lookup = (
    select(
        area_t.c.highstreet_id.label("area_id"),
        area_t.c.highstreet_name.label("area_name"),
        hex_geom_t.c.hex_id,
        overlap_fraction.label("overlap_fraction")
    )
    .join(hex_geom_t, func.ST_Intersects(hex_geom_t.c.geom, area_t.c.geom))
    .where(area_t.c.highstreet_id.in_(TEST_AREA_IDS))
).subquery("overlap_lookup")

# %%

with engine.connect() as conn:
    t_start = time.perf_counter()
    df = pd.read_sql_query(select(overlap_lookup), conn)
    print(time.perf_counter() - t_start)


# %%

single_area_select = (
    select(
        overlap_lookup.c.area_id,
        overlap_lookup.c.area_name, 
        measure_table.c.count_date,
        measure_table.c.time_indicator,
        func.sum(measure_table.c.visitor).label("total_visitors"),
        func.sum(measure_table.c.worker).label("total_workers"),
        func.sum(measure_table.c.resident).label("total_residents"),
        func.sum(measure_table.c.visitor * overlap_lookup.c.overlap_fraction).label("weighted_visitors"),
        func.sum(measure_table.c.worker * overlap_lookup.c.overlap_fraction).label("weighted_workers"),
        func.sum(measure_table.c.resident * overlap_lookup.c.overlap_fraction).label("weighted_residents")
    )
    .select_from(measure_table)
    .join(
        overlap_lookup, 
        (measure_table.c.hex_id == overlap_lookup.c.hex_id))
    .where(measure_table.c.count_date >= start_date)
    .where(measure_table.c.count_date <= func.cast(end_date + literal(1), measure_table.c.count_date.type))
    .group_by(
        overlap_lookup.c.area_id,
        overlap_lookup.c.area_name,                
        measure_table.c.count_date,
        measure_table.c.time_indicator,
    )
)


# %%

# print the full query to check if it is correct

# Compile the query with parameters injected
raw_sql = str(single_area_select.compile(
    dialect=engine.dialect, 
    compile_kwargs={"literal_binds": True}
))

# Format with clean indentation and capitalized keywords
readable_sql = sqlparse.format(raw_sql, reindent=True, keyword_case='upper')

print(readable_sql)

# %%

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Executable, ClauseElement


class ExplainAnalyze(Executable, ClauseElement):
    def __init__(self, stmt):
        self.stmt = stmt

@compiles(ExplainAnalyze, "postgresql")
def compile_explain_analyze(element, compiler, **kw):
    # This prepends the exact syntax dynamically at runtime
    return f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {compiler.process(element.stmt, **kw)}"

explain_stmt = ExplainAnalyze(single_area_select)

with engine.connect() as conn:
    # Execute and fetch the raw JSON result from Postgres
    result = conn.execute(explain_stmt)
    
    # Postgres returns the JSON as a nested array of dictionaries
    plan_json = result.scalar() 



# %%
with engine.connect() as conn:
    t_start = time.perf_counter()
    df_test = pd.read_sql_query(single_area_select, conn)
    print(time.perf_counter() - t_start)

# %%

spatial_weights_cte = (
    select(
        area_t.c.highstreet_id.label("area_id"),
        area_t.c.highstreet_name.label("area_name"),
        hex_geom_t.c.hex_id,
        overlap_fraction.label("overlap_fraction")
    )
    .join(hex_geom_t, func.ST_Intersects(hex_geom_t.c.geom, area_t.c.geom))
)


# %%
target_hex_ids = select(spatial_weights_cte.c.hex_id).distinct().scalar_subquery()

measure_t = (
        select(measure_t)
        .where(measure_table.hex_id.in_(target_hex_ids))
        .cte("heavily_filtered_measurements")
)

sw = spatial_weights_cte.c
m = measure_table

pm_reporting_date = case(
    (m.time_indicator.in_(['00-03', '03-06']), cast(m.count_date - literal(1), m.count_date.type)),
    else_=m.count_date
)

# Conditional logic mapping
am_cond = m.time_indicator.in_(['06-09', '09-12', '12-15', '15-18'])
pm_cond = m.time_indicator.in_(['18-21', '21-24', '00-03', '03-06'])

# %%

wide_aggregation_cte = (
    select(
        sw.area_id,
        sw.area_name,
        m.count_date.label("count_date"),
        pm_reporting_date.label("pm_date"),
    )
    .join(measure_t, sw.hex_id == m.hex_id)
    .group_by(sw.area_id, sw.area_name, m.count_date, pm_reporting_date)
    .limit(50)
)

with engine.connect() as conn:
    t_start = time.perf_counter()
    df = pd.read_sql_query(wide_aggregation_cte, conn)
    print(time.perf_counter() - t_start)


# %%
 
wa = wide_aggregation_cte.c

unpivot_values = """
(VALUES 
    ('resident', 'DAY', wa.count_date, wa.res_day_unw, wa.res_day_w, wa.loy_day_sum/NULLIF(wa.cnt_day,0), wa.loy_day_wsum/NULLIF(wa.wcnt_day,0), wa.dwl_day_sum/NULLIF(wa.cnt_day,0), wa.dwl_day_wsum/NULLIF(wa.wcnt_day,0)),
    ('resident', 'AM',  wa.count_date, wa.res_am_unw,  wa.res_am_w,  wa.loy_am_sum/NULLIF(wa.cnt_am,0),   wa.loy_am_wsum/NULLIF(wa.wcnt_am,0),   wa.dwl_am_sum/NULLIF(wa.cnt_am,0),   wa.dwl_am_wsum/NULLIF(wa.wcnt_am,0)),
    ('resident', 'PM',  wa.pm_date,  wa.res_pm_unw,  wa.res_pm_w,  wa.loy_pm_sum/NULLIF(wa.cnt_pm,0),   wa.loy_pm_wsum/NULLIF(wa.wcnt_pm,0),   wa.dwl_pm_sum/NULLIF(wa.cnt_pm,0),   wa.dwl_pm_wsum/NULLIF(wa.wcnt_pm,0)),
    
    ('worker',   'DAY', wa.count_date, wa.wrk_day_unw, wa.wrk_day_w, wa.loy_day_sum/NULLIF(wa.cnt_day,0), wa.loy_day_wsum/NULLIF(wa.wcnt_day,0), wa.dwl_day_sum/NULLIF(wa.cnt_day,0), wa.dwl_day_wsum/NULLIF(wa.wcnt_day,0)),
    ('worker',   'AM',  wa.count_date, wa.wrk_am_unw,  wa.wrk_am_w,  wa.loy_am_sum/NULLIF(wa.cnt_am,0),   wa.loy_am_wsum/NULLIF(wa.wcnt_am,0),   wa.dwl_am_sum/NULLIF(wa.cnt_am,0),   wa.dwl_am_wsum/NULLIF(wa.wcnt_am,0)),
    ('worker',   'PM',  wa.pm_date,  wa.wrk_pm_unw,  wa.wrk_pm_w,  wa.loy_pm_sum/NULLIF(wa.cnt_pm,0),   wa.loy_pm_wsum/NULLIF(wa.wcnt_pm,0),   wa.dwl_pm_sum/NULLIF(wa.cnt_pm,0),   wa.dwl_pm_wsum/NULLIF(wa.wcnt_pm,0)),
    
    ('visitor',  'DAY', wa.count_date, wa.vis_day_unw, wa.vis_day_w, wa.loy_day_sum/NULLIF(wa.cnt_day,0), wa.loy_day_wsum/NULLIF(wa.wcnt_day,0), wa.dwl_day_sum/NULLIF(wa.cnt_day,0), wa.dwl_day_wsum/NULLIF(wa.wcnt_day,0)),
    ('visitor',  'AM',  wa.count_date, wa.vis_am_unw,  wa.vis_am_w,  wa.loy_am_sum/NULLIF(wa.cnt_am,0),   wa.loy_am_wsum/NULLIF(wa.wcnt_am,0),   wa.dwl_am_sum/NULLIF(wa.cnt_am,0),   wa.dwl_am_wsum/NULLIF(wa.wcnt_am,0)),
    ('visitor',  'PM',  wa.pm_date,  wa.vis_pm_unw,  wa.vis_pm_w,  wa.loy_pm_sum/NULLIF(wa.cnt_pm,0),   wa.loy_pm_wsum/NULLIF(wa.wcnt_pm,0),   wa.dwl_pm_sum/NULLIF(wa.cnt_pm,0),   wa.dwl_pm_wsum/NULLIF(wa.wcnt_pm,0))
) AS v(count_type, time_window, final_date, unw_sum, w_sum, loy_unw, loy_w, dwl_unw, dwl_w)
"""

final_stmt = (
    select(
        wa.area_id,
        wa.area_name,
        literal_column("v.final_date").label("date"),
        literal_column("v.count_type").label("count_type"),
        literal_column("v.time_window").label("time_window"),
        literal_column("v.unw_sum").label("unweighted_sum"),
        literal_column("v.w_sum").label("weighted_sum"),
        literal_column("v.loy_unw").label("loyalty_unweighted_avg"),
        literal_column("v.loy_w").label("loyalty_weighted_avg"),
        literal_column("v.dwl_unw").label("dwell_time_unweighted_avg"),
        literal_column("v.dwl_w").label("dwell_time_weighted_avg")
    )
    .select_from(wide_aggregation_cte)
    .join(literal_column(unpivot_values), onclause=True) # Forces immediate execution-layer unpivoting
    .order_by(wa.area_id, literal_column("v.final_date"), literal_column("v.count_type"), literal_column("v.time_window"))
)

if limit is not None:
    final_stmt = final_stmt.limit(limit)

# Send to Database and Load into Pandas
with engine.connect() as conn:
    # Read the core statement execution cleanly straight into a DataFrame
    df = pd.read_sql_query(final_stmt, conn)