# %%
import pandas as pd 
import pymc as pm
import arviz as az
import pymc as pm
import numpy as np
import os
from pathlib import Path
import seaborn as sns
from scipy import stats 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge

from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.compose import TransformedTargetRegressor

import matplotlib.pyplot as plt

from hs_models.utils import (
    load_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
    load_9_models
)

from hs_models.constants import HEX_AREA

from dotenv import load_dotenv

from hs_models.models import AreaCountInteraction1DPartPool

load_dotenv()

sns.set_theme(style="ticks")

bucket=os.getenv("DATA_BUCKET")
if bucket is None:
    raise ValueError("DATA_BUCKET not found in .env file")

file_name=os.getenv("COUNT_DATA_FILE")
if file_name is None:
    raise ValueError("COUNT_DATA_FILE not found in .env file")

area_file=os.getenv("AREA_FILE")
if area_file is None:
    raise ValueError("AREA_FILE not found in .env file")


observation_df_filt, stats_df = load_footfall_dedupe_data(
    bucket,
    file_name,
    area_file
)

# %%

ax=(stats_df['area']/HEX_AREA).hist(bins=30)
ylim=ax.get_ylim()
ln=ax.plot((1, 1), ylim, '--')
ax.set_ylabel('# areas')
ax.set_xlabel('Area (hex units)')


# %%

# compute the extent of overcounting as a function of area 
observation_df_filt['overcount_ratio'] = observation_df_filt['worker_day'] / observation_df_filt['total_unique_workers_day'] 

count_types = ['worker', 'visitor', 'resident']
count_times = ['am', 'pm' ,'day']

for count_type in count_types:
    for count_time in count_times:
        observation_df_filt[f'overcount_ratio_{count_type}_{count_time}'] = observation_df_filt[f'{count_type}_{count_time}'] / observation_df_filt[f'total_unique_{count_type}s_{count_time}']


cols_keep = [f'overcount_ratio_{count_type}_{count_time}' for count_type in count_types for count_time in count_times] 

area_overcount_ratios = observation_df_filt[['poi_nuid', 'area'] + cols_keep].groupby('poi_nuid').mean()

for col in cols_keep:

    area_overcount_ratios[f'{col[16:]}_scale_factor'] = 1 / area_overcount_ratios[col]

area_overcount_ratios['Area (# hexes)'] = area_overcount_ratios['area'] / HEX_AREA

# %%

ax=area_overcount_ratios.plot(
    x='Area (# hexes)', 
    y='overcount_ratio_worker_day', 
    style='o', 
    logy=False, 
    logx=False, 
    ylim=(0,10), 
    xlim=(0,60),
)

ax.plot((0,50), (5,5))

plt.savefig(f'./figures/overcount/worker_day.png')


# %%

ax=area_overcount_ratios.plot(
    x='Area (# hexes)', 
    y='visitor_day_scale_factor', 
    style='o', logy=False, ylim=(0,1), xlim=(0,20))

ax.plot((0,50), (0.2, 0.2))

plt.savefig(f'./figures/overcount/visitor_day.png')


# %%

# compute the average overcount ratio in area_hexes bins from 0.5 to 5
area_overcount_ratios['area_bin'] = pd.cut(area_overcount_ratios['Area (# hexes)'], 40, labels=False, retbins=False)

area_overcount_ratios_binned = area_overcount_ratios.groupby('area_bin').mean()

area_overcount_ratios_binned['analytic_overcount_ratio'] = 4.3 + 15 / area_overcount_ratios_binned['Area (# hexes)'] ** 2

area_overcount_ratios_binned['simple_power'] = 12 / area_overcount_ratios_binned['Area (# hexes)'] ** 1.5

ax=area_overcount_ratios_binned.plot(x='Area (# hexes)', y= ['overcount_ratio_worker_day', 'analytic_overcount_ratio',], style='o')
ax.plot((0,50), (5,5))

# ax.set_ylim((0, 20))
# ax.set_xlim((0.0099, 50))

ax.set_xscale('log')
ax.set_yscale('log')

#%%

area_overcount_ratios_binned['area_2'] = 4.3 + 15 * (area_overcount_ratios_binned['Area (# hexes)'] ** (-2))


area_overcount_ratios_binned.plot(x='Area (# hexes)', y= ['overcount_ratio_worker_day', 'area_2',], style='o')


# %%

# Custom function for 1 / x^2
def inverse_square(x):
    return 1.0 / (x**2)

# Define the pipeline
pipeline = Pipeline([
    ('inv_sq', FunctionTransformer(inverse_square)),
    ('model', Ridge(alpha=0.000001))
])

X = pd.DataFrame(area_overcount_ratios['Area (# hexes)'])
y = 1 / area_overcount_ratios['overcount_ratio_worker_day']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

fig, ax = plt.subplots(1, 1)
ax.plot(X_test, y_test, 'o')
ax.plot(X_train, y_train, 'o')

# ax.plot(X_test, y_pred, 'o')

ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlim((0, 50))


# %%

from sklearn.model_selection import validation_curve, ValidationCurveDisplay

train_scores, valid_scores = validation_curve(
    pipeline, X, y, param_name="model__alpha", param_range=np.logspace(-7, 3, 3),
)

ValidationCurveDisplay.from_estimator(
   pipeline, X, y, param_name="model__alpha", param_range=np.logspace(-3, 10, 10)
)

# %%
from sklearn.metrics import mean_squared_log_error

# 1. Evaluate
error_pld = mean_squared_log_error(y_test, y_pred)

# 2. Plot error as a function of size graph


# 3. Output lookup table for this model 


# %%

# 1. Define the functional form
def model_func(x, alpha, beta, lam):
    return 1/alpha + (beta / (x**lam))

def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


# 2. Fit the model
# p0 is the initial guess for [alpha, beta, lam]
x_log = np.log(X_train['Area (# hexes)'])

p0 = [max(y_train), np.median(x_log), 1]

popt, pcov = curve_fit(sigmoid, x_log, y_train, p0=p0)

L_hat, k_hat, x0_hat = popt
print(f"Estimated parameters: L={L_hat:.3f}, k={k_hat:.3f}, x0={x0_hat:.3f}")

# 5. Generate points for a smooth line plot
x_fit = np.linspace(min(x_log), 5, 100)
y_fit = sigmoid(x_fit, *popt)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(x_log, y_train, label='Dedupe factor')
ax[0].plot(x_fit, y_fit, color='red', label=f'Sigmoid Fit: L={popt[0]:.2f}')
ax[0].set_xlabel('Log10(Area [# hexes])')
ax[0].set_ylabel('Deduplication factor')
ax[0].legend()
ax[0].set_xlim((min(x_log), max(x_log)))

ax[1].scatter(np.log(X['Area (# hexes)']), 1/y, label='Dedupe factor')
ax[1].plot(x_fit, 1/y_fit, color='red', label=f'Sigmoid Fit: L={popt[0]:.2f}')
ax[1].set_xlabel('Log10(Area [# hexes])')
ax[1].set_ylabel('Deduplication factor')
ax[1].legend()
ax[1].set_xlim((min(x_log), max(x_log)))
ax[1].set_yscale('log')


# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 1. Pick alpha by looking at median after a certain size

alpha = y_train[X_train['Area (# hexes)']>8].median()


# %%

X_train = X_train[y_train > alpha]
y_train = y_train[y_train > alpha]

# 1. Define the target transformations
def forward_transform(y):
    return np.log(y - alpha)

def inverse_transform(y_prime):
    return np.exp(y_prime) + alpha

# 2. Create the feature transformer (log(x))
# We use a Pipeline so the x-transform happens automatically during .fit()
feature_transformer = Pipeline([
    ('log_x', FunctionTransformer(np.log, check_inverse=False))
])

# 3. Wrap the Linear Regression
# We use a lambda or partial to pass the alpha parameter
model = TransformedTargetRegressor(
    regressor=QuantileRegressor(alpha=0.0001),
    func=forward_transform,
    inverse_func=inverse_transform,
)

# 4. Final Pipeline: Transform X, then Fit the Transformed Y
full_pipeline = Pipeline([
    ('prep_x', feature_transformer),
    ('regress', model)
])

full_pipeline.fit(X_train, y_train)

lambda_hat = -full_pipeline['regress'].regressor_.coef_[0]
beta_hat = np.exp(full_pipeline['regress'].regressor_.intercept_)

y_pred = alpha + beta_hat / (X_test ** lambda_hat)
y_pred_full = alpha + beta_hat / (X ** lambda_hat)

fig, ax = plt.subplots(1, 1)
ax.plot(X_test, y_test, 'o')
ax.plot(X_train, y_train, 'o')

ax.plot(X_test, y_pred, 'o')

ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_ylim((0, 20))

ax.set_xlabel('Area [# hexes]')
ax.set_ylabel('Overcount ratio')

print(f'alpha: {alpha:.2f}, beta: {beta_hat:.2f}, lambda: {lambda_hat:.2f}')

# %%

def model_func(x, alpha_in, beta_in, lambda_in):
    return alpha_in + beta_in / (x ** lambda_in)


# 1. Create a temporary DataFrame for calculation
calc_df = X.copy()
calc_df['y'] = y
calc_df['y_pred'] = model_func(calc_df['Area (# hexes)'], alpha, beta_hat, lambda_hat)
calc_df['residual'] = calc_df['y'] - calc_df['y_pred']
calc_df['abs_residual'] = np.abs(calc_df['y'] - calc_df['y_pred'])
calc_df['pct_residual'] = (calc_df['y'] - calc_df['y_pred']) / calc_df['y']
calc_df['pct_abs_residual'] = np.abs(calc_df['y'] - calc_df['y_pred']) / calc_df['y']

# 2. Calculate the Squared Error for every row
calc_df['sq_error'] = (calc_df['y'] - calc_df['y_pred'])**2

calc_df['pct_error'] = np.sqrt((calc_df['y'] - calc_df['y_pred'])**2) / calc_df['y']

# 3. Create bins for X (e.g., 5 equal-width bins)
calc_df['X_bins'], bins = pd.qcut(calc_df['Area (# hexes)'], q=np.linspace(0,1,20), retbins=True)

# 4. Group by bins and calculate the Mean of the Squared Errors
binned_mse = calc_df.groupby('X_bins')['sq_error'].mean()
binned_pce = calc_df.groupby('X_bins')['pct_error'].mean()
binned_pcar = calc_df.groupby('X_bins')['pct_abs_residual'].mean()
binned_pcr = calc_df.groupby('X_bins')['pct_residual'].median()

bin_midpoints = calc_df['X_bins'].apply(lambda x: x.mid).cat.categories

print(binned_pce)

fig, ax = plt.subplots(1, 1)
plt.plot(bin_midpoints, 100*binned_pcr, 'o')

ax.set_xscale('log')
# ax.set_yscale('log')

ax.set_xlabel('Area [# hexes]')
ax.set_ylabel('median % residual of overcount ratio')

grouped=calc_df.groupby('X_bins')


# %%
ax=calc_df.plot(x='Area (# hexes)', y='residual', style='o')
ax.set_ylim((0.001,6000))
ax.plot((0,50), (0,0), 'k--')
ax.set_yscale('log')
ax.set_xscale('log')

# %%
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Pick alpha by looking at median after a certain size
alpha = y_train[X_train['Area (# hexes)']>5].median()

def model_func_power_law(x, alpha_in, beta_in, lambda_in):
    return alpha_in + beta_in / (x ** lambda_in)


# B-spline with 4 + 3 - 1 = 6 basis functions
model = make_pipeline(SplineTransformer(n_knots=4, degree=5), Ridge(alpha=1e-240))
model.fit(X_train, y_train)

y_pred_spline = model.predict(X_test)

fig, ax = plt.subplots(1, 1)
ax.plot(X_test, y_test, 'o')
ax.plot(X_train, y_train, 'o')

ax.plot(X_test, y_pred_spline, 'o')

ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_ylim((0, 50))

ax.set_xlabel('Area [# hexes]')
ax.set_ylabel('Overcount ratio')


# %%

fig, ax = plt.subplots(1, 3, figsize=(20,6))

area_bounds = [0, 2, 11, 100]

for idx, ub in enumerate(area_bounds[1:]):
        
    test_mask = (X_test['Area (# hexes)']<=ub) & (X_test['Area (# hexes)']> area_bounds[idx])
    train_mask = (X_train['Area (# hexes)']<=ub) & (X_train['Area (# hexes)']> area_bounds[idx])

    ax[idx].plot(X_test[test_mask], y_test[test_mask], 'o')
    ax[idx].plot(X_train[train_mask], y_train[train_mask], 'o')

    # ax[idx].set_xscale('log')
    # ax[idx].set_yscale('log')
    ax[idx].set_ylim((0, 20))


# ax[idx].set_xlabel('Area [# hexes]')
# ax[idx].set_ylabel('Overcount ratio')

# %%

from scipy.interpolate import UnivariateSpline

# s=0 forces the spline through every point; 
# Increase s to smooth out noise
spline = UnivariateSpline(X_train, y_train, s=0)

# Calculate the first derivative
dy_dx = spline.derivative()(X_train)
y_s = spline(X_train)

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].plot(X_train, y_s, 'o')
ax[1].plot(X_train, dy_dx, 'o')

ax[0].set_xlabel('Area [# hexes]')
ax[0].set_ylabel('Overcount ratio')

# %%

# 1. Create a temporary DataFrame for calculation
calc_df = X.copy()
calc_df['y'] = y
calc_df['y_pred'] = model_func(calc_df['Area (# hexes)'], alpha, beta_hat, lambda_hat)
calc_df['residual'] = calc_df['y'] - calc_df['y_pred']
calc_df['abs_residual'] = np.abs(calc_df['y'] - calc_df['y_pred'])
calc_df['pct_residual'] = (calc_df['y'] - calc_df['y_pred']) / calc_df['y']
calc_df['pct_abs_residual'] = np.abs(calc_df['y'] - calc_df['y_pred']) / calc_df['y']

# 2. Calculate the Squared Error for every row
calc_df['sq_error'] = (calc_df['y'] - calc_df['y_pred'])**2

calc_df['pct_error'] = np.sqrt((calc_df['y'] - calc_df['y_pred'])**2) / calc_df['y']

# 3. Create bins for X (e.g., 5 equal-width bins)
calc_df['X_bins'], bins = pd.qcut(calc_df['Area (# hexes)'], q=np.linspace(0,1,20), retbins=True)

# 4. Group by bins and calculate the Mean of the Squared Errors
binned_mse = calc_df.groupby('X_bins')['sq_error'].mean()
binned_pce = calc_df.groupby('X_bins')['pct_error'].mean()
binned_pcar = calc_df.groupby('X_bins')['pct_abs_residual'].mean()
binned_pcr = calc_df.groupby('X_bins')['pct_residual'].median()

bin_midpoints = calc_df['X_bins'].apply(lambda x: x.mid).cat.categories

print(binned_pce)

fig, ax = plt.subplots(1, 1)
plt.plot(bin_midpoints, 100*binned_pcr, 'o')

ax.set_xscale('log')
# ax.set_yscale('log')

ax.set_xlabel('Area [# hexes]')
ax.set_ylabel('median % residual of overcount ratio')

grouped=calc_df.groupby('X_bins')