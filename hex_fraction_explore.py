# %% imports and setup

import geopandas as gpd
import pandas as pd
from datetime import date
import seaborn as sns
import hs_models.utils as util 
from hs_models.constants import FOOTFALL_COUNTS_TABLE

import lightgbm as lgb
import optuna

sns.set_theme(style="ticks")
sns.set_style('darkgrid')

# %% Load data

df_long = util.load_data_long(interaction_cols=['area_hex', 'duplicated'])

# %% Load data

df_hex = util.get_hex_geometry()
df_hs, df_tc, df_bid = util.get_area_geometry_from_database()

# %% Load data
db_metadata, _ = util.get_db_metadata()

footfall_table = db_metadata.tables[FOOTFALL_COUNTS_TABLE]

for column in footfall_table.columns:
    print(f"Column: {column.name:<20} | Type: {column.type}")


# %% Load footfall count data
query_dict = {
    "select": [
        "hex_id", 
        "count_date", 
        "day", 
        "time_indicator", 
        "resident", 
        "worker", 
        "visitor", 
        "loyalty_percentage",
        "dwell_time"],
    "limit": 20
}

df_footfall = util.get_footfall_data(query_dict)

# %% Load footfall count data aggregated

start_date = date.fromisoformat('2024-01-01')
end_date = date.fromisoformat('2025-08-16')

df_footfall_agg = util.get_summed_hex_data_custom_areas(limit=50, start_date=start_date, end_date=end_date)

# %% Load footfall count data aggregated

df_footfall_agg.loc[df_footfall_agg['count_type'] == 'visitor', ['count_date', 'count_type', 'time_window', 'unweighted_count_sum', 'weighted_count_sum']].head(13)

# %% Load footfall count data aggregated

df_long.loc[df_long['poi_name'] == 'Chapel Market, Islington', ['count_date', 'count_type', 'time_indicator', 'duplicated', 'total_unique']].head(12)

# %% fetch best hyperparams

study_name="lgb-linear-tree"
hyperparam_db = "sqlite:///hyperparams.sqlite3"

study = optuna.create_study(
    direction="minimize",
    storage=hyperparam_db,  # Specify the storage URL here.
    study_name=study_name,
    load_if_exists=True
    )

# Access the best parameters and value
best_params = study.best_params
best_value = study.best_value

# %% retrain on full data

y = df_long['total_unique']
X = df_long.drop(columns=['poi_name', 'poi_id', 'poi_uid', 'total_unique', 'count_date'])

d_full = lgb.Dataset(X, label=y)

best_params['objective'] = 'regression'
best_params['metric'] = 'mape'

bst_final = lgb.train(
    best_params,
    d_full,
)

final_predictions = bst_final.predict(X)

score = util.evaluate_dedupe_mape_threshold(y, final_predictions)

g = util.error_vs_area_plot(X, y, final_predictions)

# %%


