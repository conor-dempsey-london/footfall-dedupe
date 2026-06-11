# %% Imports and setup
import pandas as pd 
import numpy as np
from dotenv import load_dotenv
import seaborn as sns
import hs_models.utils as util 
from hs_models.models import Hill2DRegressor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklego.meta import GroupedPredictor

import lightgbm as lgb

load_dotenv()

sns.set_theme(style="ticks")
sns.set_style('darkgrid')

# %% Load data

df_long = util.load_data_long()

df_long = df_long[df_long['count_type'] != 'visitors']

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

y = df_long['total_unique']

X = df_long.drop(columns=['poi_name', 'poi_id', 'poi_uid', 'total_unique', 'count_date'])


# %% Define sklearn models

hill2d_model = Hill2DRegressor()
hill2d_3x3 = GroupedPredictor(Hill2DRegressor(), groups=['count_type', 'time_indicator'])

preprocessor = ColumnTransformer(
    transformers=[
        (
            'interact', 
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), 
            ['duplicated', 'avg_dwell_time', 'area_hex'],
        ),
    ],
    remainder='drop'
)

linreg_global = Pipeline([
    ('prep', preprocessor),
    ('regressor', LinearRegression())
])

linreg_3x3 = GroupedPredictor(linreg_global, groups=['count_type', 'time_indicator'])

skmodels = {
    'linear regression': linreg_global, 
    'linear regression 3x3': linreg_3x3, 
    'Hill2D regression': hill2d_model, 
    # 'Hill2D 3x3': hill2d_3x3
    }

# %% Define and fit LGB models

fold_scores = []
models = []

gkf = GroupKFold(n_splits=5)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=X['poi_nuid']))
X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

scores = {}
print('Fitting sk models')
for model_name, skmodel in skmodels.items():
    skmodel.fit(X_train, y_train)
    y_pred = skmodel.predict(X_val)
    scores[model_name] = util.evaluate_dedupe_mape_threshold(y_val, y_pred)
    g = util.error_vs_area_plot(X_val, y_val, y_pred, area_col='area_hex')
    g.savefig(f'./figures/error_analysis/area/{model_name}.png')



print('Fitting LGB model')
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=X['poi_nuid'])):
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    params = {'objective': 'regression', 'metric': 'mape', 'verbosity': -1}
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

    fold_scores.append(score)
    models.append(bst)

scores['LGB'] = np.mean(fold_scores)


# %% Model comparison


# Error vs area
for model_name, skmodel in skmodels.items():
    y_pred = skmodel.predict(X_val)
    g = util.error_vs_area_plot(X_val, y_val, y_pred, area_col='area_hex')
    g.savefig(f'./figures/error_analysis/area/{model_name}.png')

pred_test   = bst.predict(X_val)
g = util.error_vs_area_plot(X.iloc[val_idx], y_val, pred_test)
g.savefig('./figures/error_analysis/area/linear_tree.png')

pred_train   = bst.predict(X_train)
g = util.error_vs_area_plot(X_train, y_train, pred_train)
g.savefig('./figures/error_analysis/area/linear_tree_train.png')

score_train = util.evaluate_dedupe_mape_threshold(y_train, pred_train)
scores['LGB train'] = score_train

# Model score comparison
print(scores)

# Examples plots across models


# Error vs time


# Feature important for Linear tree model
lgb.plot_importance(bst)


# %%




