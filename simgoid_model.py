# %%
import pandas as pd 
import numpy as np
import os
from pathlib import Path
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt

import piecewise_regression

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge, LinearRegression, QuantileRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from hs_models.utils import (
    load_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
    load_9_models
)

from hs_models.constants import HEX_AREA

from dotenv import load_dotenv

load_dotenv()

sns.set_theme(style="ticks")
sns.set_style('darkgrid')

bucket=os.getenv("DATA_BUCKET")
file_name=os.getenv("COUNT_DATA_FILE")
area_file=os.getenv("AREA_FILE")

observation_df_filt, stats_df = load_footfall_dedupe_data(
    bucket,
    file_name,
    area_file
)

observation_df_filt.dropna(inplace=True)

count_types = ['worker', 'resident', 'visitor']
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

def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

X = pd.DataFrame(area_overcount_ratios['Area (# hexes)'])
y = area_overcount_ratios['worker_day_scale_factor']
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.4, shuffle=True)

x_log_train = np.log(X_train['Area (# hexes)'])
x_log_test = np.log(X_test['Area (# hexes)'])

obs_test = observation_df_filt[observation_df_filt['poi_nuid'].isin(X_test.index)]
obs_test['area [hex]'] = obs_test['area'] / HEX_AREA

obs_test = obs_test[
    ['poi_nuid', 'area [hex]', 'area'] 
    + [f'{count_type}_{count_time}' for count_type in count_types for count_time in count_times] 
    + [f'total_unique_{count_type}s_{count_time}' for count_type in count_types for count_time in count_times]]

areas_lookup = np.logspace(np.log10(0.02), np.log10(10), 2000)

scale_dfs = []

for count_type in count_types[:-1]:
    for count_time in count_times:

        df = pd.DataFrame()

        y_train = (
            area_overcount_ratios.loc[X_train.index, f'{count_type}_{count_time}_scale_factor']
        )

        y_test = (
            area_overcount_ratios.loc[X_test.index, f'{count_type}_{count_time}_scale_factor']
        )

        # Fit the model
        # p0 is the initial guess for [alpha, beta, lam]
        p0 = [max(y_train), np.median(x_log_train), 1]

        popt, pcov = curve_fit(sigmoid, x_log_train, y_train, p0=p0)

        L_hat, k_hat, x0_hat = popt
        print(f"Estimated parameters: L={L_hat:.3f}, k={k_hat:.3f}, x0={x0_hat:.3f}")

        # 5. Generate points for a smooth line plot
        x_fit = np.linspace(min(x_log_train), 5, 100)
        y_fit = sigmoid(x_fit, *popt)

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(x_log_test, y_test, label='Dedupe factor')
        ax[0].plot(x_fit, y_fit, color='red', label=f'sigmoid fit: L={popt[0]:.2f}')
        ax[0].set_xlabel('Area [# hexes]')
        ax[0].set_ylabel('Deduplication factor')
        ax[0].set_ticks()
        ax[0].set_xticklabels(np.round(np.exp(ax[0].get_xticks()), 1))
        ax[0].legend()
        ax[0].set_xlim((min(x_log_test), max(x_log_test)))
        # ax[0].set_ylim((-0.01, 0.32))

        ax[1].scatter(x_log_test, 1/y_test)
        ax[1].plot(x_fit, 1/y_fit, color='red')
        ax[1].set_xlabel('Area [# hexes]')
        ax[1].set_ylabel('Overcount ratio')
        ax[1].set_xticklabels(np.round(np.exp(ax[1].get_xticks()), 1))
        ax[1].set_xlim((min(x_log_train), max(x_log_train)))
        ax[1].set_yscale('log')

        fig.suptitle(f'{count_type} {count_time}')

        plt.savefig(f'./figures/logistic/dedupe_factor_{count_type}_{count_time}.png')

        scale_factors = sigmoid(np.log(areas_lookup/HEX_AREA), *popt)

        df['Area (km2)'] = areas_lookup
        df['scale factor'] = scale_factors
        df['count_type'] = count_type
        df['count_time'] = count_time

        scale_dfs.append(df)

        obs_test.loc[:, f'model_scale_factor_{count_type}_{count_time}']  = sigmoid(np.log(obs_test['area']/HEX_AREA), *popt)

        obs_test[f'predicted_{count_type}_{count_time}'] = obs_test[f'{count_type}_{count_time}'] * obs_test[f'model_scale_factor_{count_type}_{count_time}']

        obs_test[f'pct_residual_{count_type}_{count_time}'] = 100*(obs_test[f'total_unique_{count_type}s_{count_time}'] - obs_test[f'predicted_{count_type}_{count_time}']) / obs_test[f'total_unique_{count_type}s_{count_time}']

        obs_test[f'pct_residual_{count_type}_{count_time}'] = 100*(obs_test[f'total_unique_{count_type}s_{count_time}'] - obs_test[f'predicted_{count_type}_{count_time}']) / obs_test[f'total_unique_{count_type}s_{count_time}']

        obs_test_grouped = obs_test.groupby('poi_nuid').mean()

        obs_test_grouped['X_bins'], bins = pd.qcut(obs_test_grouped['area']/HEX_AREA, q=np.linspace(0,1,10), retbins=True)

        fig, ax = plt.subplots(1, 1)
        obs_test_grouped.plot(
            x='area [hex]', 
            y=f'pct_residual_{count_type}_{count_time}', 
            style='o', 
            ax=ax, 
            label='true count - predicted count')

        ax.set_xscale('log')
        # ax.set_yscale('log')

        ax.set_xlabel('Area [# hexes]')
        ax.set_ylabel('% residual of deduplicated count')
        ax.plot((0, 50), (0,0), 'k--')
        # ax.plot((1, 1), (-130,25), 'k--')

        # ax.set_xlim((0.1, 50))
        ax.set_ylim((-500, 100))
        ax.get_legend().remove()

        fig.suptitle(f'{count_type} {count_time}')

        plt.savefig(f'./figures/logistic/{count_type}_{count_time}_residual_by_area.png')

        obs_test_grouped['area binned [# hexes]'] = obs_test_grouped['X_bins'].apply(lambda x: x.mid)

        fig, ax = plt.subplots(1, 1)

        sns.lineplot(
            obs_test_grouped, 
            x='area binned [# hexes]', 
            y=f'pct_residual_{count_type}_{count_time}', 
            errorbar=('ci', 95),
            markers=True,
            estimator=np.median,
        )

        ylim=(-160,42)
        plt.ylim(ylim)
        plt.xscale('log')
        plt.ylabel('% residual count')
        plt.legend(['estimated median', '95% C.I.'], loc='lower right')
        # plt.plot((1, 1), ylim, '--k')

        plt.savefig(f'./figures/logistic/median_residual_CI_{count_type}_{count_time}.png')


scale_factor_df = pd.concat(scale_dfs)

scale_factor_df.to_csv('./output/scale_factor_lookup_logistic.csv')

# %%

    