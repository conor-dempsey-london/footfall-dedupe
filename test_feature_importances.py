# %% Imports and setup
import pandas as pd 
import seaborn as sns
import hs_models.utils as util 

import joblib

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

import lightgbm as lgb
import optuna

from supertree import SuperTree

from dotenv import load_dotenv
load_dotenv()

sns.set_theme(style="ticks")
sns.set_style('darkgrid')

# Load data
df_long = util.load_data_long()

df_hs, df_tc, df_bid = util.join_pois_from_db(df_long)

df_long['dup_zscored_by_area'] = df_long.groupby('poi_nuid')['duplicated'].transform(util.zscore)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

interaction_cols = ['area_hex', 'duplicated']
interactions = poly.fit_transform(df_long[interaction_cols])

joblib.dump(poly, './models/poly_interactions_areahex_dup.pkl')

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


# %% define inputs and outputs and train

y = df_long['total_unique']
X_full = df_long.drop(columns=[
    'poi_name', 
    'poi_id',
    'poi_uid', 
    'count_date',
    'total_unique', 
])
X = df_long.drop(columns=[
    'poi_name', 
    'poi_id', 
    'poi_uid', 
    'total_unique', 
    'count_date',
    'area',
    'dup_zscored_by_area',
    'caz_inner_outer',
    'timestamp',
    'day_of_week',
    'month',
    'year',
    'hour',
    'is_weekend',
    'day',
    ])

# Define and fit LGB models
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=X['poi_nuid']))
X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

X_train = X_train.drop(columns=['poi_nuid'])
X_val_poinuid = X_val['poi_nuid']
X_val   = X_val.drop(columns=['poi_nuid'])

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

X_train.to_csv('./data/X_train.csv')
X_val.to_csv('./data/X_test.csv')

# train model without area ids
bst_no_id = lgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=15)]
)

# %% train model with area ids

X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

bst = lgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=15)]
)

bst.save_model('./models/lgb_minimal_with_ids.txt')

# %% train model without area ids and all features

X_train_full, X_val_full = X_full.iloc[train_idx], X_full.iloc[test_idx]

X_train_full = X_train_full.drop(columns=['poi_nuid'])
X_val_poinuid_full = X_val_full['poi_nuid']
X_val_full   = X_val_full.drop(columns=['poi_nuid'])

dtrain = lgb.Dataset(X_train_full, label=y_train)
dval = lgb.Dataset(X_val_full, label=y_val, reference=dtrain)

bst_full_noid = lgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=15)]
)

# Evaluate on the unseen areas
preds_noid_full = bst_full_noid.predict(X_val_full)
preds_train_noid_full = bst_full_noid.predict(X_train_full)

score_noid_full = util.evaluate_dedupe_mape_threshold(y_val, preds_noid_full)
score_train_noid_full = util.evaluate_dedupe_mape_threshold(y_train, preds_train_noid_full)

g, error_df_noid_full = util.error_vs_area_plot(X_val, y_val, preds_noid_full)
ax=g.axes[0][0]
ax.text(
    0.6, 
    0.9, 
    s=f'score test: {score_noid_full:.2f}\nscore train: {score_train_noid_full:.2f}', 
    transform=ax.transAxes,
    ha='left')

g.savefig('./figures/error_analysis/area/lightgbm_noid_full.png')

# %% train model with area ids and all features

X_train_full, X_val_full = X_full.iloc[train_idx], X_full.iloc[test_idx]

dtrain = lgb.Dataset(X_train_full, label=y_train)
dval = lgb.Dataset(X_val_full, label=y_val, reference=dtrain)

bst_full = lgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=15)]
)

# Evaluate on the unseen areas
preds_full = bst_full.predict(X_val_full)
preds_train_full = bst_full.predict(X_train_full)

score_full = util.evaluate_dedupe_mape_threshold(y_val, preds_full)
score_train_full = util.evaluate_dedupe_mape_threshold(y_train, preds_train_full)

g, error_df_full = util.error_vs_area_plot(X_val_full, y_val, preds_full)
ax=g.axes[0][0]
ax.text(
    0.6, 
    0.9, 
    s=f'score test: {score_full:.2f}\nscore train: {score_train_full:.2f}', 
    transform=ax.transAxes,
    ha='left')

g.savefig('./figures/error_analysis/area/lightgbm_full.png')


# %% look at test error when trained without area ids

# Evaluate on the unseen areas
preds_noid = bst_no_id.predict(X_val.drop(columns='poi_nuid'))
preds_train_noid = bst_no_id.predict(X_train.drop(columns='poi_nuid'))

score_noid = util.evaluate_dedupe_mape_threshold(y_val, preds_noid)
score_train_noid = util.evaluate_dedupe_mape_threshold(y_train, preds_train_noid)

g, error_df_noid = util.error_vs_area_plot(X_val, y_val, preds_noid)
ax=g.axes[0][0]
ax.text(
    0.6, 
    0.9, 
    s=f'score test: {score_noid:.2f}\nscore train: {score_train_noid:.2f}', 
    transform=ax.transAxes,
    ha='left')

g.savefig('./figures/error_analysis/area/lightgbm_minimal_noid.png')

# %% look at test error when trained with area ids

# Evaluate on the unseen areas
preds = bst.predict(X_val)
preds_train = bst.predict(X_train)

score = util.evaluate_dedupe_mape_threshold(y_val, preds)
score_train = util.evaluate_dedupe_mape_threshold(y_train, preds_train)

g, error_df = util.error_vs_area_plot(X_val, y_val, preds)
ax=g.axes[0][0]
ax.text(
    0.6, 
    0.9, 
    s=f'score test: {score:.2f}\nscore train: {score_train:.2f}', 
    transform=ax.transAxes,
    ha='left')

g.savefig('./figures/error_analysis/area/lightgbm_minimal.png')


# %%

print((
    f'score w id and minimal features: {score:.2f}\n'
    f'score without id and minimal features: {score_noid:.2f}\n'
    f'score with id and full features: {score_full:.2f}\n'
    f'score without id and full features: {score_noid_full:.2f}\n'
))

# %%
df = pd.DataFrame([
    {'area id used in training?': 'No', 'all features': score_full, 'minimal features': score},
    {'area id used in training?': 'Yes', 'all feautres': score_noid_full, 'minimal features': score_noid},
])

df.index=['without area id', 'with area id']

util.plot_df(df, './figures/models/scores_trees.png', title_string='model scores (MAPE)')

# %%

model_full = lgb.LGBMRegressor()

model_full._Booster = bst_full
model_full._n_features = bst_full.num_feature()
model_full._objective = bst_full.dump_model()["objective"]
model_full.fitted_ = True

r = permutation_importance(model_full, X_val_full, y_val, n_repeats=30, random_state=0)

# %%
for i in r.importances_mean.argsort()[::-1]:
    print(f"{X_train_full.columns[i]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")
        

# %%

sorted_importances_idx = r.importances_mean.argsort()
importances = pd.DataFrame(
    r.importances[sorted_importances_idx].T,
    columns=X_val_full.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in error")
ax.figure.tight_layout()

ax.figure.savefig('./figures/models/lightgbm_full_feature_importances.png')

# %% Visualise the tree

st = SuperTree(
    bst, 
    X, 
    y, 
    X.columns.to_list(),
    "deduplicated count"
)

# Visualize the tree
st.show_tree(which_tree=0)

# %%

df_full=X_val_full.drop(columns=['area'])[[
    'poi_nuid',
    'area_hex duplicated',
    'duplicated',
    'area_hex',
    'count_type',
    'time_indicator',
    'dup_zscored_by_area',
    'is_weekend',
    'caz_inner_outer',
    'avg_dwell_time',
    'poi_type',
]].sample(5, axis=0)

df_full.rename(columns={'dup_zscored_by_area': 'dup_util.zscored'}, inplace=True)

df=X_val.sample(5, axis=0)

# %%

df = pd.DataFrame([
    {'area id used in training?': 'No', 'all features': score_full, 'minimal features': score},
    {'area id used in training?': 'Yes', 'all features': score_noid_full, 'minimal features': score_noid},
])

df.index=['without area id', 'with area id']

util.plot_df(df, './figures/models/scores_trees.svg', title_string='model scores (MAPE)')

# %%


util.plot_df(df_full, "./figures/data/X_val_full.png")
# %%

util.plot_df(df, "./figures/data/X_val.png")



# %%
