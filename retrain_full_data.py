# %% Imports and setup
import pandas as pd 
import seaborn as sns
import hs_models.utils as util 
import joblib

from sklearn.preprocessing import PolynomialFeatures

import lightgbm as lgb
import optuna

from dotenv import load_dotenv
load_dotenv()

sns.set_theme(style="ticks")
sns.set_style('darkgrid')

# Load data
df_long = util.load_data_long()

df_hs, df_tc, df_bid = util.join_pois_from_db(df_long)

# feature engineering
def zscore(s):
    # Handle groups with no variance to avoid NaNs
    if s.std() == 0:
        return 0
    return (s - s.mean()) / s.std()

df_long['dup_zscored_by_area'] = df_long.groupby('poi_nuid')['duplicated'].transform(zscore)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

interaction_cols = ['area_hex', 'duplicated']
interactions = poly.fit_transform(df_long[interaction_cols])

interaction_names = poly.get_feature_names_out(interaction_cols)
df_interactions = pd.DataFrame(interactions, columns=interaction_names, index=df_long.index)

df_long = pd.concat([df_long, df_interactions.drop(columns=interaction_cols)], axis=1)

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

# %% save final model

joblib.dump(poly, './models/poly_interactions.pkl')

bst_final.save_model('./models/ldt.txt')

# save final model




# %%
