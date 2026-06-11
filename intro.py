# %%
import pandas as pd 
import numpy as np
from dotenv import load_dotenv
import os
import seaborn as sns
from hs_models.constants import HEX_AREA

from scipy.optimize import curve_fit

from sklego.meta import GroupedPredictor

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import random

# %%

load_dotenv()

sns.set_theme(style="ticks")
sns.set_style('darkgrid')

file_name=os.getenv("COUNT_DATA_FILE")
area_file=os.getenv("AREA_FILE")

df_long = pd.read_csv(file_name)
area_df = pd.read_csv(area_file) 
area_df['area'] = area_df['area'] / 1e6

df_long['poi_nuid']=df_long['poi_type'] + '_' + df_long['poi_id'].astype(str)
area_df['poi_nuid'] = area_df['poi_type'] + '_' + area_df['poi_id'].astype(str)
area_df = area_df.groupby('poi_nuid')['area'].first().reset_index()

df_long.rename(columns={
    'resident': 'duplicated_residents',
    'worker': 'duplicated_workers',
    'visitor': 'duplicated_visitors',
    'total_unique_domestic_visitors': 'total_unique_visitors'
}, inplace=True)

# add area data to main df
df_long = df_long.merge(area_df[['poi_nuid', 'area']], on='poi_nuid', how='inner')

df_long["id"] = df_long.index

df_long = pd.wide_to_long(
    df_long,
    stubnames=["total_unique", "duplicated"],
    i="id",
    j="count_type",
    sep='_',
    suffix=r"\w+"
)

df_long = df_long.reset_index().drop(columns=['id'])
df_long.dropna(inplace=True)

# %%
df_long['area [hex]'] = df_long['area'] / HEX_AREA

df_long = df_long[[
    'poi_id', 'poi_type', 'poi_nuid',
    'count_date', 'area', 'area [hex]', 'avg_dwell_time', 'caz_inner_outer', 
    'count_type', 'time_indicator', 'total_unique', 'duplicated'
]]

all_pois = df_long['poi_nuid'].unique().tolist()
pois_test = random.sample(all_pois, round(0.2*len(all_pois)))

df_test = df_long.loc[df_long['poi_nuid'].isin(pois_test)]
df_train = df_long.loc[~df_long['poi_nuid'].isin(pois_test)]

cols_input = ['poi_nuid', 'area [hex]', 'count_type', 'time_indicator', 'duplicated']
cols_predict = ['total_unique']

X_test, y_test = df_test[cols_input], df_test[cols_predict]
X_train, y_train = df_train[cols_input], df_train[cols_predict]

# %%


def evaluate_dedupe_mape_threshold(y_test, y_pred):
    m = y_test >= 100
    return mean_absolute_percentage_error(y_test[m], y_pred[m])

# set up evaluation function 
def evaluate_dedupe(
        data,
        true_deduped_col='total_unique', 
        deduped_col='model_total_unique',
        poi_col='poi_nuid', 
        type_col='count_type',
        time_col='time_indicator',
    ):

    # to avoid emphasising performance only for large areas we evaluate
    # by looking at metrics computed at the level of each poi and then averaged
    poi_grouped = (data
                   .groupby(poi_col)[[deduped_col, true_deduped_col]]
                   .mean()
    )

    # MAPE across all data
    mape_all = evaluate_dedupe_mape_threshold(poi_grouped[true_deduped_col], poi_grouped[deduped_col])

    # MAPE for each combination of type x time
    # mape_grouped = poi_grouped.groupby([type_col, time_col])

    return mape_all



def bundle_test_pred(df_test, y_pred):
    df_test['model_total_unique'] = y_pred
    return df_test

# %%

model_scores = {}

# baseline model - fit a line to the entire dataset 
reg = LinearRegression().fit(X_train[['duplicated']], y_train['total_unique'])

y_pred = reg.predict(X_test[['duplicated']])

fig, ax = plt.subplots()
df_test.plot(x='duplicated', y='total_unique', style='o', ax=ax)
xls = ax.get_xlim()
yls = ax.get_ylim()
plt.plot(xls, reg.intercept_ + reg.coef_*xls)

mape_all = evaluate_dedupe(
    bundle_test_pred(df_test, y_pred)
)

# model_scores['linear regression pooled'] = 


# %%

# Sort and compute gradients along each axis
df = df_long.loc[df_long['count_type'] != 'visitors', ['area [hex]', 'total_unique', 'duplicated']].sort_values(['duplicated', 'area [hex]'])

# Bin the 'area' variable
df['area_bin'] = pd.qcut(df['area [hex]'], q=40)
df['dup_bin'] = pd.qcut(df['duplicated'], q=20)

# For each area bin, estimate d(total_count)/d(duplicated)
with np.errstate(divide='ignore', invalid='ignore'):
    partials_dup = (
        df.sort_values('duplicated')
        .groupby('area_bin')
        .apply(lambda g: np.mean(np.isfinite(np.gradient(g['total_unique'].values, g['duplicated'].values))))
    )

    partials_a = (
        df.sort_values('area [hex]')
        .groupby('dup_bin')
        .apply(lambda g: np.mean(np.isfinite(np.gradient(g['total_unique'].values, g['area [hex]'].values))))
    )

def hill_model(x, x0, n):
    return (x**n) / (x0**n + x**n)

def logistic_model(a, L, a0, k):
    return L / (1 + (a0 / a)**k)

params_pa, _ = curve_fit(hill_model, partials_a.index.categories.mid[:-1], partials_a[:-1], p0=[1500, 1])
params_pd, _ = curve_fit(hill_model, partials_dup.index.categories.mid, partials_dup, p0=[5, 0.5])

fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].plot(partials_a.index.categories.mid, partials_a, 'o')
ax[0].plot(partials_a.index.categories.mid, hill_model(partials_a.index.categories.mid, *params_pa))
ax[0].set_xlabel('Duplicated count')
ax[0].set_ylabel('Rate of change of true count wrt area')
ax[0].set_xlim((-3000, 100000))
# ax[0].set_ylim((0, 1.2))

pd_plot = np.linspace(partials_dup.index.categories.mid.min(), partials_dup.index.categories.mid.max(), 1000)
ax[1].plot(partials_dup.index.categories.mid, partials_dup, 'o')
ax[1].plot(pd_plot, hill_model(pd_plot, *params_pd))
ax[1].set_xlabel('Area [hex]')
ax[1].set_ylabel('Rate of change of true count wrt duplicated count')

# %%

df = df.loc[df['duplicated']<=100000, ]

def bin_diff(df, count_col='total_unique'):
    # 1. Define the number of bins (e.g., 20x20 grid)
    num_bins_dup = 15
    num_bins_area = 5

    # 2. Find quantile edges for each axis
    dup_edges = np.quantile(df['duplicated'], np.linspace(0, 1, num_bins_dup + 1))
    a_edges = np.quantile(df['area [hex]'], np.linspace(0, 1, num_bins_area + 1))

    # 3. Use pd.cut to assign each row to a bin
    df['dup_bin'] = pd.cut(df['duplicated'], bins=dup_edges, include_lowest=True)
    df['area_bin'] = pd.cut(df['area [hex]'], bins=a_edges, include_lowest=True)

    # 4. Aggregate: calculate the mean of count_col for each bin
    binned_data = df.groupby(['dup_bin', 'area_bin'])[count_col].mean().unstack()
    binned_data.columns = binned_data.columns.categories.mid
    binned_data.index = binned_data.index.categories.mid
    binned_data.columns.name = 'area'
    binned_data.index.name = 'duplicated'

    binned_data_long = binned_data.reset_index().melt(id_vars='duplicated', var_name='area', value_name='count')

    # Create the coordinate grid using the uneven edges
    dup_grid, area_grid = np.meshgrid(dup_edges, a_edges)

    # 1. Calculate bin centers (midpoints between edges)
    dup_centers = (dup_edges[:-1] + dup_edges[1:]) / 2
    area_centers = (a_edges[:-1] + a_edges[1:]) / 2

    # 2. Create the meshgrid using centers, not edges
    dup_c, area_c = np.meshgrid(dup_centers, area_centers)

    # 3. Plot using the centers
    plt.figure(figsize=(7, 5))
    # dup_c, area_c, and binned_data.T.values will now all be (num_bins_area, num_bins_dup)
    contour = plt.contourf(dup_c, area_c, binned_data.T.values, levels=20, cmap='magma')
    plt.colorbar(contour, label='Total Unique')

    # Add contour lines for better definition
    plt.contour(dup_c, area_c, binned_data.T.values, levels=10, colors='white', alpha=0.3)

    plt.xlabel('Duplicated')
    plt.ylabel('Area')
    # plt.xscale('log')
    plt.yscale('log')
    plt.show()

    # Calculate partial derivatives
    # Note: np.gradient returns [dZ/dy, dZ/dx] because axis 0 is rows (dup) 
    # and axis 1 is columns (area)
    dc_dd, dc_da = np.gradient(binned_data.values, dup_centers, area_centers)

    dc_dd_frame = pd.DataFrame(
        dc_dd, 
        index=binned_data.index, 
        columns=binned_data.columns
    )
    dc_dd_long = dc_dd_frame.reset_index().melt(id_vars='duplicated', var_name='area', value_name='dc_dd')

    dc_da_frame = pd.DataFrame(
        dc_da, 
        index=binned_data.index, 
        columns=binned_data.columns
    )
    dc_da_long = dc_da_frame.reset_index().melt(id_vars='duplicated', var_name='area', value_name='dc_da')

    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    contour = ax[0].contourf(dup_c, area_c, dc_da.T, levels=20, cmap='magma')
    plt.colorbar(contour, label='Rate of change of true count wrt area')
    ax[0].contour(dup_c, area_c, dc_da.T, levels=10, colors='white', alpha=0.3)

    ax[0].set_xlabel('Duplicated')
    ax[0].set_ylabel('Area')
    # plt.xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_title('dc/da')

    contour = ax[1].contourf(dup_c, area_c, dc_dd.T, levels=20, cmap='magma')
    plt.colorbar(contour, label='Rate of change of true count wrt dup')
    ax[1].contour(dup_c, area_c, dc_dd.T, levels=10, colors='white', alpha=0.3)

    ax[1].set_xlabel('Duplicated')
    ax[1].set_ylabel('')
    # plt.xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title('dc/dd')

    return binned_data_long, dc_da_long, dc_dd_long


# %%

binned_data_long, dc_da_long, dc_dd_long = bin_diff(df)

# %%
df = df.join(df_long[['poi_nuid', 'time_indicator', 'count_type']])

count_types = ['workers', 'residents']
count_times = ['DAY', 'PM', 'AM']

results_list = []
for count_type, count_time in [(x,y) for x in count_types for y in count_times]:
    results = {}
    results['data'], results['dc_da'], results['dc_dd'] = bin_diff(df[
        (df['count_type']==count_type) & (df['time_indicator']==count_time)])
    results_list.append(results)


# %%

fig, ax = plt.subplots(1, 2, figsize=(12,5))
sns.lineplot(data=binned_data_long, x='duplicated', y='count', hue='area', ax=ax[0], hue_norm=LogNorm())
sns.lineplot(data=binned_data_long, x='area', y='count', hue='duplicated', ax=ax[1])
ax[0].set_ylabel('count')
ax[1].set_ylabel('')
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')

# %%

fig, ax = plt.subplots(1, 2, figsize=(12,5))
sns.lineplot(data=binned_data_long, x='duplicated', y='count', hue='area', ax=ax[0], hue_norm=LogNorm())
sns.lineplot(data=binned_data_long, x='area', y='count', hue='duplicated', ax=ax[1])
ax[0].set_ylabel('count')
ax[1].set_ylabel('')


# %%

fig, ax = plt.subplots(2, 2, figsize=(12,10))
sns.lineplot(data=dc_da_long, x='duplicated', y='dc_da', hue='area', ax=ax[0][0], hue_norm=LogNorm())
sns.lineplot(data=dc_da_long, x='area', y='dc_da', hue='duplicated', ax=ax[0][1])
ax[0][0].set_ylabel('dc/da')
ax[0][1].set_ylabel('')

sns.lineplot(data=dc_dd_long, x='duplicated', y='dc_dd', hue='area', ax=ax[1][0], hue_norm=LogNorm())
sns.lineplot(data=dc_dd_long, x='area', y='dc_dd', hue='duplicated', ax=ax[1][1])
ax[1][0].set_ylabel('dc/dd')
ax[1][1].set_ylabel('')
# ax[1][1].set_xscale('log')

# %%
area_info = df[['area [hex]', 'poi_nuid', 'area_bin']].groupby('poi_nuid').first().reset_index()


# fit a 2D Hill function

# 1. Define the 2D Hill Model with Half-Saturation Constants
def hill_2d_centered(coords, L, x0, a0, n, k):
    x, a = coords
    # Normalize each variable by its respective half-saturation constant
    term = ((x / x0)**n) * ((a / a0)**k)
    return L * term / (1 + term)

# 2. Example Data Setup
# a ranges from 0.2 to 30; x ranges from 500 to 60,000
x_data = df['duplicated'].values # Duplicated counts
a_data = df['area [hex]'].values # Area
y_observed = df['total_unique'].values # Unique counts

# Initial guesses [L, x0, a0, n, k] are critical for non-linear convergence
p0 = [55000, 30000, 15, 1, 1.2]

# Use bounds to ensure parameters like x0 and a0 remain positive
bounds = (0, [np.inf, np.inf, np.inf, 10, 10])

popt, _ = curve_fit(hill_2d_centered, (x_data, a_data), y_observed, p0=p0, bounds=bounds)

# 4. Results
L_fit, x0_fit, a0_fit, n_fit, k_fit = popt
print(f"L (Max): {L_fit:.1f}")
print(f"x0 (Midpoint x): {x0_fit:.1f}")
print(f"a0 (Midpoint a): {a0_fit:.1f}")
print(f"n (Steepness x): {n_fit:.2f}")
print(f"k (Steepness a): {k_fit:.2f}")

# %%

# predicted values
df['predicted_hill2d'] = hill_2d_centered((x_data, a_data), *popt)

binned_data_long_pred, dc_da_long_pred, dc_dd_long_pred = bin_diff(df, count_col='predicted_hill2d')

# %%
def fit_predict_hill2d_group(group):    
    p0 = [55000, 30000, 15, 1, 1.2]

    # Use bounds to ensure parameters like x0 and a0 remain positive
    bounds = (0, [np.inf, np.inf, np.inf, 10, 10])
    x, a, y = group['duplicated'].values, group['area [hex]'].values, group['total_unique'].values
    popt, _ = curve_fit(hill_2d_centered, (x, a), y, p0=p0, bounds=bounds)
    group['predicted'] =  hill_2d_centered((x, a), *popt)

    return group['predicted']


df['predicted_hill2d_3x3'] = df.groupby(['count_type', 'time_indicator'], group_keys=False).apply(fit_predict_hill2d_group)

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# %%


class Hill2DRegressor(BaseEstimator, RegressorMixin):
    def __init__(
            self, 
            area_col='area_hex',
            dup_col='duplicated',
            p0=[55000, 30000, 15, 1, 1.2],
            bounds=(0, [200000, 50000, 100, 10, 10])):
        self.p0 = p0
        self.bounds = bounds
        self.area_col   = area_col
        self.dup_col    = dup_col

    @staticmethod
    def _hill_2d_centered(coords, L, x0, a0, n, k):
        x, a = coords
        # Normalize each variable by its respective half-saturation constant
        term = ((x / x0)**n) * ((a / a0)**k)
        return L * term / (1 + term)

    def fit(self, X, y):
        X = X[[self.dup_col, self.area_col]]
        # Validate inputs
        X, y = check_X_y(X, y)
        x, a = X[:, 0], X[:, 1]
        self.popt, _ = curve_fit(hill_2d_centered, (x, a), y, p0=p0, bounds=bounds)
        # Store metadata for compatibility
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = X[[self.dup_col, self.area_col]]
        X = check_array(X)
        x, a = X[:, 0], X[:, 1]
        # Return mean for all predictions
        y_pred = hill_2d_centered((x, a), *self.popt)
        return y_pred  

    def score(self, X, y):
        check_is_fitted(self)
        return evaluate_dedupe_mape_threshold(y, self.predict(X)) 

# %%
hill2d_model = Hill2DRegressor(area_col='area [hex]')
hill2d_model.fit(X_train[X_train['duplicated']<=100000], y_train[X_train['duplicated']<=100000])

# %%
y_pred = hill2d_model.predict(X_test)
score_hill2d = evaluate_dedupe_mape_threshold(y_test, y_pred)

# %%

hill2d_3x3 = GroupedPredictor(Hill2DRegressor(area_col='area_hex'), groups=['count_type', 'time_indicator'])
hill2d_3x3.fit(X_train, y_train)
y_pred = hill2d_3x3.predict(X_test)
score_hill2d3x3 = evaluate_dedupe_mape_threshold(y_test, y_pred)

# %%


def fit_hill2d_group(group):    
    p0 = [55000, 30000, 15, 1, 1.2]

    # Use bounds to ensure parameters like x0 and a0 remain positive
    bounds = (0, [np.inf, np.inf, np.inf, 10, 10])
    x, a, y = group['duplicated'].values, group['area [hex]'].values, group['total_unique'].values
    popt, _ = curve_fit(hill_2d_centered, (x, a), y, p0=p0, bounds=bounds)

    return popt


hill2d_model_params = df.groupby(['count_type', 'time_indicator'], group_keys=False).apply(fit_hill2d_group)


# %%
from scipy.stats import linregress

def fit_predict_linreg_group(group):    

    res = linregress(group['duplicated'], group['total_unique'])
    group['predicted'] =  res.intercept + res.slope * group['duplicated']

    return group['predicted']

df['predicted_linreg_3x3'] = df.groupby(['count_type', 'time_indicator'], group_keys=False).apply(fit_predict_linreg_group)

import statsmodels.api as sm
import statsmodels.formula.api as smf

def fit_predict_linreg_a_group(group):
    # 1. Define independent (X) and dependent (y) variables
    X = group[['duplicated', 'area [hex]']]
    
    # 2. Explicitly add a constant to include an intercept
    X = sm.add_constant(X) 
    y = group['total_unique']
    
    # 3. Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y, X).fit()

    group['predicted'] =  (
        model.params['const'] + 
        model.params['duplicated'] * group['duplicated'] + 
        model.params['area [hex]'] * group['area [hex]']
    )

    return group['predicted']

df['predicted_linreg_3x3_a'] = df.groupby(['count_type', 'time_indicator'], group_keys=False).apply(fit_predict_linreg_a_group)


def fit_predict_linreg_a_group_inter(group):
    group.rename(columns={'area [hex]': 'area'}, inplace=True)

    model = smf.ols(formula='total_unique ~ duplicated * area', data=group).fit()
    
    group['predicted']=  model.predict(group)
    return group['predicted']

df['predicted_linreg_3x3_a_interact'] = df.groupby(['count_type', 'time_indicator'], group_keys=False).apply(fit_predict_linreg_a_group_inter)



# %%

fig, ax = plt.subplots(2, 2, figsize=(12,10))
sns.lineplot(data=binned_data_long, x='duplicated', y='count', hue='area', ax=ax[0][0], hue_norm=LogNorm())
sns.lineplot(data=binned_data_long, x='area', y='count', hue='duplicated', ax=ax[0][1])
ax[0][0].set_ylabel('count')
ax[0][1].set_ylabel('')

sns.lineplot(data=binned_data_long_pred, x='duplicated', y='count', hue='area', ax=ax[1][0], hue_norm=LogNorm())
sns.lineplot(data=binned_data_long_pred, x='area', y='count', hue='duplicated', ax=ax[1][1])
ax[1][0].set_ylabel('count')
ax[1][1].set_ylabel('')

fig.savefig('./figures/hill/count_vs_area_dup_model_pred.png')

# %%

df['predicted_linreg'] = reg.predict(df[['duplicated']])

# %%

def sigmoid(row):
    return row['L'] / (1 + np.exp(-row['k'] * (np.log(row['duplicated']) - row['x0'])))

sig_model_params = pd.read_csv('./output/sigmoid_model_params.csv')
sig_model_params['count_time'] = sig_model_params['count_time'].str.upper()
sig_model_params['count_type'] = sig_model_params['count_type'].astype(str) + 's'

df = df.merge(sig_model_params, left_on=['count_type', 'time_indicator'], right_on=['count_type', 'count_time'])

df['scale_factor_sig'] = df.apply(sigmoid, axis=1)

df['predicted_sigmoid'] = df['duplicated'] * df['scale_factor_sig']

df.drop(columns=['L', 'k', 'x0_log', 'x0', 'count_time', 'scale_factor_sig'], inplace=True)

mape_sigmoid = evaluate_dedupe(
    df[df['total_unique']>100],
    deduped_col='predicted_sigmoid',
)

# %%

mape_linreg = evaluate_dedupe(
    df[df['total_unique']>100],
    deduped_col='predicted_linreg',
)

mape_linreg_3x3 = evaluate_dedupe(
    df[df['total_unique']>100],
    deduped_col='predicted_linreg_3x3',
)

mape_sigmoid = evaluate_dedupe(
    df[df['total_unique']>100],
    deduped_col='predicted_sigmoid',
)

mape_hill2d = evaluate_dedupe(
    df[df['total_unique']>100],
    deduped_col='predicted_hill2d',
)

mape_hill2d_33 = evaluate_dedupe(
    df[df['total_unique']>100],
    deduped_col='predicted_hill2d_3x3',
)

mape_linreg_a = evaluate_dedupe(
    df[df['total_unique']>100],
    deduped_col='predicted_linreg_3x3_a',
)

mape_linreg_a_int = evaluate_dedupe(
    df[df['total_unique']>100],
    deduped_col='predicted_linreg_3x3_a_interact',
)

mape_linreg_unpooled = evaluate_dedupe(
    df[df['total_unique']>100],
    deduped_col='predicted_unpooled_lin_reg',
)

print(f'MAPE lin. reg.: {mape_linreg:.2f}, MAPE lin. reg. 3x3: {mape_linreg_3x3:.2f}, MAPE lin. reg. with area: {mape_linreg_a:.2f}, MAPE lin. reg. with area interaction: {mape_linreg_a_int:.2f}, MAPE sigmoid 3x3: {mape_sigmoid:.2f}, MAPE 2D Hill: {mape_hill2d:.2f}, MAPE 2D Hill 3x3: {mape_hill2d_33:.2f}, MAPE lin. reg. unpooled: {mape_linreg_unpooled:.2f}')


# %%

dfg = (
    df.groupby(['poi_nuid', 'count_type', 'time_indicator'])[['total_unique', 'duplicated', 'predicted_hill2d_3x3', 'predicted_hill2d', 'predicted_linreg', 'predicted_linreg_3x3', 'predicted_sigmoid', 'predicted_linreg_3x3_a', 'predicted_linreg_3x3_a_interact']]
    .mean()
    .reset_index()
    .merge(area_info, how='left', on='poi_nuid')
)

dfg.loc[dfg['area [hex]']<0.5, 'size_flag'] = 'tiny'
dfg.loc[(dfg['area [hex]']>=0.5)&(dfg['area [hex]']<1), 'size_flag'] = 'small'
dfg.loc[(dfg['area [hex]']>=1)&(dfg['area [hex]']<2), 'size_flag'] = 'one-ish'
dfg.loc[(dfg['area [hex]']>=2)&(dfg['area [hex]']<10), 'size_flag'] = 'medium'
dfg.loc[dfg['area [hex]']>=10, 'size_flag'] = 'large'

# %%

def add_size_flag(df):
    df.loc[df['area [hex]']<0.5, 'size_flag'] = 'tiny'
    df.loc[(df['area [hex]']>=0.5)&(df['area [hex]']<1), 'size_flag'] = 'small'
    df.loc[(df['area [hex]']>=1)&(df['area [hex]']<2), 'size_flag'] = 'one-ish'
    df.loc[(df['area [hex]']>=2)&(df['area [hex]']<10), 'size_flag'] = 'medium'
    df.loc[df['area [hex]']>=10, 'size_flag'] = 'large'

    return df


# %%

g = sns.lmplot(
    data=dfg, 
    x='total_unique', 
    y='predicted_hill2d',
    row='count_type',
    col='size_flag',
    col_order=['tiny', 'small', 'one-ish', 'medium', 'large'],
    hue='time_indicator',
    facet_kws = {
        'sharex': False,
        'sharey': False
    }
)

for ax in g.axes.flat:
    ax.axline((0, 0), slope=1, color='k', ls='--')
    ax.grid(True, axis='both', ls=':')


# %%

# for large areas, group by duplicated and plot dup vs count
eg_pois = random.sample(
    df.loc[
        df['area_bin'] == df['area_bin'].unique()[0], 
        'poi_nuid'].unique().tolist(), 
    9)

eg_pois_small = [
    'highstreets_354', 'towncentres_217', 'highstreets_398',
    'highstreets_260', 'highstreets_293', 'highstreets_310',
    'towncentres_142', 'towncentres_74', 'towncentres_59'
]

eg_pois_large = [
    'highstreets_605', 'highstreets_333', 'bids_55',
    'bids_47', 'highstreets_520', 'towncentres_27',
    'towncentres_41', 'highstreets_508', 'bids_26'
]

# %%
cols_keep = [
        'area [hex]', 
        'total_unique', 
        'duplicated', 
        'predicted_hill2d_3x3',
        'predicted_hill2d',
        'predicted_linreg',
        'predicted_sigmoid',
        'predicted_linreg_3x3_a_interact',
        'predicted_unpooled_lin_reg', 
        'poi_nuid',
        'count_type',
        'time_indicator'
    ]

df_eg_large = df.loc[
    (df['poi_nuid'].isin(eg_pois_large)) & (df['time_indicator'].isin(['DAY', 'PM'])), 
    cols_keep
]

df_eg_small = df.loc[
    (df['poi_nuid'].isin(eg_pois_small)) & (df['time_indicator'].isin(['DAY', 'PM'])), 
    cols_keep
]

# %%

# 1. Create a dictionary applying 'mean' to all columns except the mode column and groups
agg_dict = {col: 'mean' for col in df.columns if col not in ['poi_nuid', 'area_bin' ,'time_indicator', 'count_type', 'dup_bin']}

# 2. Explicitly add the mode logic for your specific column
# We use .iloc[0] to grab the first mode if multiple exist
agg_dict['dup_bin'] = lambda x: x.mode().iloc[0] if not x.mode().empty else None

# 3. Apply the aggregation
result = df.groupby(['poi_nuid', 'area_bin' ,'time_indicator', 'count_type']).agg(agg_dict).reset_index()

# %%
sns.set_theme(font_scale=1.5)

g = sns.relplot(
    data=df_eg_small.sample(frac=0.5, random_state=22),
    x='duplicated',
    y='total_unique',
    col='poi_nuid',
    hue='count_type',
    style='time_indicator', 
    col_wrap=3, 
    s=200, 
    alpha=0.8,
)

g.map_dataframe(
    sns.scatterplot, 
    x='duplicated', 
    y='predicted_unpooled_lin_reg', 
    color='black', 
    alpha=0.4, 
)

g.map_dataframe(
    sns.lineplot, 
    x='duplicated', 
    y='predicted_linreg', 
    color='black', 
    alpha=0.4, 
)

g.set(ylim=(-100, 5000))

g.set_axis_labels("", "")
g.axes[3].set_ylabel("BT deduplicated count")    
g.axes[7].set_xlabel("Duplicated count")

g.savefig('./figures/eda/examples_linreg_unpooled_small.png')


# %%

df['abs_pct_error_hill2d_33'] = 100*np.abs((df['total_unique'] - df['predicted_hill2d_3x3'])) / df['total_unique']
df['abs_pct_error_hill2d'] = 100*np.abs((df['total_unique'] - df['predicted_hill2d'])) / df['total_unique']
df['abs_pct_error_sigmoid'] = 100*np.abs((df['total_unique'] - df['predicted_sigmoid'])) / df['total_unique']
df['abs_pct_error_linreg_unpooled'] = 100*np.abs((df['total_unique'] - df['predicted_unpooled_lin_reg'])) / df['total_unique']
df['abs_pct_error_lgb'] = 100*np.abs((df['total_unique'] - df['predicted_lgb'])) / df['total_unique']

df_error = df[['poi_nuid', 'area [hex]', 'count_type', 'time_indicator', 'abs_pct_error_hill2d_33', 'abs_pct_error_sigmoid', 'abs_pct_error_hill2d', 'abs_pct_error_linreg_unpooled', 'abs_pct_error_lgb']]
df_error = df_error[np.isfinite(df_error['abs_pct_error_hill2d_33'])]
df_error = df_error[np.isfinite(df_error['abs_pct_error_hill2d'])]
df_error = df_error[np.isfinite(df_error['abs_pct_error_sigmoid'])]
df_error = df_error[np.isfinite(df_error['abs_pct_error_linreg_unpooled'])]
df_error = df_error[np.isfinite(df_error['abs_pct_error_lgb'])]

poi_error = df_error.groupby(['poi_nuid', 'count_type', 'time_indicator'])[['area [hex]', 'abs_pct_error_sigmoid', 'abs_pct_error_hill2d_33', 'abs_pct_error_hill2d', 'abs_pct_error_linreg_unpooled', 'abs_pct_error_lgb']].mean().reset_index()

# Identify which columns to keep as they are
id_cols = ['poi_nuid', 'time_indicator', 'count_type', 'area [hex]']

# Melt the two y-variables into one 'value' column
df_melted = poi_error.melt(
    id_vars=id_cols, 
    value_vars=['abs_pct_error_sigmoid', 'abs_pct_error_hill2d_33', 'abs_pct_error_hill2d', 'abs_pct_error_linreg_unpooled', 'abs_pct_error_lgb'], 
    var_name='metric_type',               
    value_name='measurement'                 
)


# %%
# error vs area
g=sns.relplot(
    data=poi_error,
    x='area [hex]',
    y='abs_pct_error_hill2d_33',
    col='count_type',     
    hue='time_indicator', 
)
g.set(yscale='log')
# g.set(xscale='log')

g.axes[0][0].set_ylabel('Mean absolute % error')
g.axes[0][0].set_xlabel('Area [hexes]')
g.axes[0][0].set_title('Residents')
g.axes[0][1].set_title('Workers')
g.axes[0][1].set_xlabel('Area [hexes]')
g.legend.set_title('Time')

# %%

def error_vs_area_plot(df, col_pred, fig_name, col_true='total_unique'):

    df['error'] = 100*np.abs((df[col_true] - df[col_pred])) / df[col_true]

    df_error = df[[
        'poi_nuid', 
        'area [hex]', 
        'count_type', 
        'time_indicator', 
        'error', 
    ]]

    df_error = df_error[np.isfinite(df_error['error'])]

    poi_error = df_error.groupby(['poi_nuid', 'count_type', 'time_indicator'])[['area [hex]', 'error']].mean().reset_index()

    poi_error['area_bin'] = pd.qcut(poi_error['area [hex]'], q=40)
    poi_error['area_bin'] = poi_error['area_bin'].apply(lambda x: x.mid)

    g=sns.relplot(
        data=poi_error,
        x='area_bin',
        y='error',
        hue='time_indicator',
        kind='line',     
        errorbar='ci',   
        estimator='median',
        facet_kws = {'sharey': False}
    )
    g.refline(y=15, color='red', linestyle='--', linewidth=1, label='15% error')
    g.refline(y=30, color='red', linestyle='--', linewidth=1, label='30% error')

    g.set(yscale='log')
    g.set(xscale='log')
    g.set(ylim=(3, 1000))
    g.legend.set_title('Time')
    g.axes[0][0].set_xlabel('Area [hexes]')
    g.axes[0][0].set_ylabel('Avg. absolute % error')

    g.savefig(f'./figures/eda/{fig_name}.png')


# %%

g = sns.relplot(
    data=df_eg_large,
    x='duplicated',
    y='total_unique',
    row='count_type',
    col='time_indicator',
    # hue='poi_nuid'
)

# g.savefig('./figures/eda/examples.png')


# %%
# fit lines to each poi in 3x3 groups

from scipy import stats

def get_regression(group):
    # Returns slope and intercept; requires at least 2 points
    if len(group) < 2:
        return pd.Series({'slope': None, 'intercept': None})
    
    slope, intercept, _, _, _ = stats.linregress(group['duplicated'], group['total_unique'])
    return pd.Series({'slope': slope, 'intercept': intercept})

# 1. Group by the identifier and the two categorical columns
# 2. Apply the regression function
# 3. Use reset_index() to return to long form
results_linreg_df = (
    df.groupby(['poi_nuid', 'time_indicator', 'count_type'])
    .apply(get_regression, include_groups=False)
    .reset_index()
).merge(area_info, on='poi_nuid')

# %%
df=df.merge(results_linreg_df, on=['poi_nuid', 'count_type', 'time_indicator'])

# %%

df['predicted_unpooled_lin_reg'] = df['intercept'] + df['slope'] * df['duplicated']


# %%

# Visualises the overall spread of slopes, colored by one grouping factor
sns.displot(data=results_linreg_df, x='slope', hue='count_type', kind='kde', fill=True)

# %%
# Shows variability and outliers across the 9 combinations
sns.boxplot(data=results_linreg_df, x='count_type', y='slope', hue='time_indicator')

# %%
# Shows the correlation between area size and slope with marginal distributions
sns.jointplot(data=results_linreg_df, x='area [hex]', y='slope', kind='reg')

# %%
# Relationship between size and slope, separating the 9 combinations
sns.relplot(
    data=results_linreg_df, 
    x='area [hex]', y='slope', 
    hue='count_type', style='time_indicator',
    alpha=0.6 # Useful if areas overlap
)


# %%

# df = add_size_flag(df)

import lightgbm as lgb
from sklearn.model_selection import GroupKFold


y = df_long['total_unique']
X = df_long[[
    'duplicated', 
    'time_indicator',
    'count_type',
    'area [hex]',
    'poi_nuid', 
    'avg_dwell_time',
    'caz_inner_outer',
    'poi_type'
]]

X.rename(columns={'area [hex]': 'area_hex'}, inplace=True)
X['count_type'] = X['count_type'].astype('category')
X['caz_inner_outer'] = X['caz_inner_outer'].astype('category')
X['poi_type'] = X['poi_type'].astype('category')
X['time_indicator'] = X['time_indicator'].astype('category')
# X['size_flag'] = X['size_flag'].astype('category')
X['poi_nuid'] = X['poi_nuid'].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

area_counts = X_train['poi_nuid'].value_counts()
weights = X_train['poi_nuid'].map(lambda x: 1.0 / area_counts[x])

train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'mape',
    'linear_tree': True
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    valid_names=['training', 'validation'],
    callbacks=[lgb.early_stopping(stopping_rounds=15)]
)

gkf = GroupKFold(n_splits=5)

fold_scores = []
models = []

fold_scores_h = []
models_h = []

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

    # hill2d_3x3.fit(X_train)
    # preds_h = hill2d_3x3.predict(X_val)
    # score_h = evaluate_dedupe_mape_threshold(y_val, preds_h)
    # fold_scores_h.append(score)
    # models_h.append(bst)
    
    # Evaluate on the unseen areas
    preds = bst.predict(X_val)
    score = evaluate_dedupe_mape_threshold(y_val, preds)
    fold_scores.append(score)
    models.append(bst)
    
    print(f"Fold {fold} MAPE: {score:.2f}")

print(f"mean MAPE: {np.mean(fold_scores):.4f}")
# print(f"mean MAPE Hill 2D 3x3: {np.mean(fold_scores_h):.4f}")

y_pred = bst.predict(X_test)
y_pred_all = bst.predict(X)

df_long['predicted_lgb'] = y_pred_all

lgb.plot_importance(bst)

# %%
# Export the model to a dictionary
lgb.plot_tree(bst, tree_index=0)


# %%

# Average predictions from all models in your 'models' list
all_preds = [model.predict(X) for model in models]
final_pred = np.mean(all_preds, axis=0)

df_long['predicted_lgb_groupval_mean'] = final_pred


# %%

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_dedupe_mape_threshold(y_test, y_pred):
    m = (y_test >= 100) & (np.isfinite(y_pred))
    return mean_absolute_percentage_error(y_test[m], y_pred[m])

pred_cols = [
    'predicted_unpooled_lin_reg', 'predicted_linreg',  'predicted_sigmoid',
    'predicted_linreg_3x3', 'predicted_linreg_3x3_a', 'predicted_linreg_3x3_a_interact', 
    'predicted_hill2d', 'predicted_hill2d_3x3', 'predicted_lgb', 'predicted_lgb_groupval_mean']

model_names = [
    'Unpooled lin. reg.', 'pooled lin. reg', 'sigmoid scale factor',
    '3x3 lin. regs.', '3x3 lin. regs. with area', '3x3 lin. regs. w area interaction',
    'Hill 2D pooled', '3x3 Hill 2D', 'Light GBM', 'LGBM group'
]

results = []
for idx, pred_col in enumerate(pred_cols):
    result={}
    y_pred = df_long.loc[y_test.index, pred_col]
    m = ~np.isnan(y_pred)
    result['mae'] = mean_absolute_error(y_test[m], y_pred[m])
    result['rmse'] = np.sqrt(mean_squared_error(y_test[m], y_pred[m]))
    result['r2'] = r2_score(y_test[m], y_pred[m])
    result['mape [thresholded]'] = evaluate_dedupe_mape_threshold(y_test[m], y_pred[m])
    results.append(result)

# error_vs_area_plot(df, col_pred=pred_col, fig_name=f'{model_names[idx]}_error_vs_area')

results = pd.DataFrame(results, index=pred_cols)
# %%

error_vs_area_plot(df, col_pred=pred_col, fig_name=f'{model_names[idx]}_error_vs_area')


# %%

lgb.plot_split_value_histogram(bst, feature='area_hex')