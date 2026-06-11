# %% Imports and setup
import pandas as pd 
import seaborn as sns
import hs_models.utils as util 

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures

import lightgbm as lgb
import optuna

from dotenv import load_dotenv
load_dotenv()

sns.set_theme(style="ticks")
sns.set_style('darkgrid')

# %% Load data

df_long = util.load_data_long()

df_hs, df_tc, df_bid = util.join_pois_from_db(df_long)

# %% feature engineering
def zscore(s):
    # Handle groups with no variance to avoid NaNs
    if s.std() == 0:
        return 0
    return (s - s.mean()) / s.std()

df_long['dup_zscored_by_area'] = df_long.groupby('poi_nuid')['duplicated'].transform(zscore)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

target_cols = ['area_hex', 'duplicated']
interactions = poly.fit_transform(df_long[target_cols])

interaction_names = poly.get_feature_names_out(target_cols)
df_interactions = pd.DataFrame(interactions, columns=interaction_names, index=df_long.index)

df_long = pd.concat([df_long, df_interactions.drop(columns=target_cols)], axis=1)

# %% Setup training and testing data 

def objective(trial):
    y = df_long['total_unique']

    X = df_long.drop(columns=['poi_name', 'poi_id', 'poi_uid', 'total_unique', 'count_date'])

    # Define and fit LGB models
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=X['poi_nuid']))
    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    params = {
        'objective': 'regression', 
        'metric': 'mape', 
        'verbosity': -1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=15)]
    )
    
    # Evaluate on the unseen areas
    preds = bst.predict(X_val)
    score = util.evaluate_dedupe_mape_threshold(y_val, preds)

    return score



# %% Run optuna loop

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

# %%

study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

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
