# %%
import pandas as pd 
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

import random

from sqlalchemy import select, MetaData, Table, create_engine
from scipy import stats
import datetime as dt

from sklearn.model_selection import train_test_split

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from hs_models.utils import (
    load_footfall_dedupe_data, 
)

from hs_models.constants import HEX_AREA

from dotenv import load_dotenv

load_dotenv()

sns.set_theme(style="ticks")
sns.set_style('darkgrid')

file_name=os.getenv("COUNT_DATA_FILE")
area_file=os.getenv("AREA_FILE")

observation_df_filt, stats_df = load_footfall_dedupe_data(
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
    area_overcount_ratios[f'scale_factor_{col[16:]}'] = 1 / area_overcount_ratios[col]

area_overcount_ratios['Area (# hexes)'] = area_overcount_ratios['area'] / HEX_AREA

observation_df_filt['area [hex]'] = observation_df_filt['area'] / HEX_AREA


# %%

def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

X = pd.DataFrame(area_overcount_ratios['Area (# hexes)'])
y = area_overcount_ratios['scale_factor_worker_day']
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.4, shuffle=True)

x_log_train = np.log(X_train['Area (# hexes)'])
x_log_test = np.log(X_test['Area (# hexes)'])

obs_test = observation_df_filt[observation_df_filt['poi_nuid'].isin(X_test.index)]

obs_test = obs_test[
    ['poi_nuid', 'area [hex]', 'area', 'count_date'] 
    + [f'{count_type}_{count_time}' for count_type in count_types for count_time in count_times] 
    + [f'total_unique_{count_type}s_{count_time}' for count_type in count_types for count_time in count_times]]

areas_lookup = np.logspace(np.log10(0.02), np.log10(10), 2000)

scale_dfs = []

model_params = []

for count_type in count_types:
    for count_time in count_times:

        df = pd.DataFrame()

        y_train = (
            area_overcount_ratios.loc[X_train.index, f'scale_factor_{count_type}_{count_time}']
        )

        y_test = (
            area_overcount_ratios.loc[X_test.index, f'scale_factor_{count_type}_{count_time}']
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
        ax[0].set_xticks([-1, 0, 1, 2, 3])
        ax[0].set_xticklabels(np.round(np.exp([-1, 0, 1, 2, 3]), 1))
        ax[0].legend()
        ax[0].set_xlim((min(x_log_test), max(x_log_test)))
        # ax[0].set_ylim((-0.01, 0.32))

        ax[1].scatter(x_log_test, 1/y_test)
        ax[1].plot(x_fit, 1/y_fit, color='red')
        ax[1].set_xlabel('Area [# hexes]')
        ax[1].set_ylabel('Overcount ratio')
        ax[1].set_xticks([-1, 0, 1, 2, 3])
        ax[1].set_xticklabels(np.round(np.exp([-1, 0, 1, 2, 3]), 1))
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

        L, k, x0 = popt

        model_params.append(
            {
                'L': L,
                'k': k,
                'x0_log': x0,
                'x0': np.exp(x0),
                'count_type': count_type,
                'count_time': count_time
            }
        )

        obs_test.loc[:, f'scale_factor_model_{count_type}_{count_time}']  = sigmoid(np.log(obs_test['area']/HEX_AREA), *popt)

        obs_test[f'predicted_{count_type}_{count_time}'] = obs_test[f'{count_type}_{count_time}'] * obs_test[f'scale_factor_model_{count_type}_{count_time}']

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

        plt.savefig(f'./figures/error_analysis/{count_type}_{count_time}_residual_by_area.png')

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

        plt.savefig(f'./figures/error_analysis/median_residual_CI_{count_type}_{count_time}.png')


scale_factor_df = pd.concat(scale_dfs)

scale_factor_df.to_csv('./output/scale_factor_lookup_logistic.csv', index=False)

model_params = pd.DataFrame(model_params)

model_params.to_csv('./output/sigmoid_model_params.csv', index=False)

# %%

unique_highstreets = obs_test.loc[obs_test['poi_nuid'].str.startswith('highstreets'), 'poi_nuid'].str.slice(12).unique().astype(int).tolist()

unique_tcs = obs_test.loc[obs_test['poi_nuid'].str.startswith('towncentres'), 'poi_nuid'].str.slice(12).unique().astype(int).tolist()

unique_bids = obs_test.loc[obs_test['poi_nuid'].str.startswith('bids'), 'poi_nuid'].str.slice(5).unique().astype(int).tolist()

database = os.getenv("PG_DATABASE")
username = os.getenv("PG_USER")
password = os.getenv("PG_PASSWORD")
host = os.getenv("PG_HOST")
port = os.getenv("PG_PORT")

engine = create_engine(
    f"postgresql+psycopg2://{username}:{password}@"
    f"{host}:{port}/{database}"
)

schema = 'gisapdata'

hs_table_name = 'regen_high_streets_proposed_2'
tc_table_name = 'planning_town_centre_all_2024'
bid_table_name = 'regen_business_improvement_districts_27700_live'

metadata = MetaData()
hs_table = Table(hs_table_name, metadata, autoload_with=engine)
tc_table = Table(tc_table_name, metadata, autoload_with=engine)
bid_table = Table(bid_table_name, metadata, autoload_with=engine)

query_hs = select(hs_table).where(hs_table.c.highstreet_id.in_(unique_highstreets))
query_tc = select(tc_table).where(tc_table.c.tc_id.in_(unique_tcs))
query_bid = select(bid_table).where(bid_table.c.bid_id.in_(unique_bids))

with engine.connect() as conn:
    df_hs   = pd.read_sql(query_hs, conn)
    df_tc   = pd.read_sql(query_tc, conn)
    df_bid  = pd.read_sql(query_bid, conn)

# %%

for count_type, count_time in [(x,y) for x in count_types for y in count_times]:

    sigmoid_params = model_params.loc[(model_params['count_type'] == count_type) & (model_params['count_time'] == count_time), ['L', 'k', 'x0']].iloc[0].tolist()

    observation_df_filt.loc[:, f'scale_factor_model_{count_type}_{count_time}']  = sigmoid(np.log(observation_df_filt['area [hex]']), *sigmoid_params)

    observation_df_filt[f'predicted_{count_type}_{count_time}'] = observation_df_filt[f'{count_type}_{count_time}'] * observation_df_filt[f'scale_factor_model_{count_type}_{count_time}']

    observation_df_filt[f'pct_residual_{count_type}_{count_time}'] = 100*(observation_df_filt[f'total_unique_{count_type}s_{count_time}'] - observation_df_filt[f'predicted_{count_type}_{count_time}']) / observation_df_filt[f'total_unique_{count_type}s_{count_time}']

    observation_df_filt[f'pct_residual_{count_type}_{count_time}'] = 100*(observation_df_filt[f'total_unique_{count_type}s_{count_time}'] - observation_df_filt[f'predicted_{count_type}_{count_time}']) / observation_df_filt[f'total_unique_{count_type}s_{count_time}']

    observation_df_filt.loc[
        :, f'scale_factor_true_{count_type}_{count_time}'
    ] = 1 / observation_df_filt.loc[:, f'overcount_ratio_{count_type}_{count_time}']

observation_df_filt['weekday'] = observation_df_filt.count_date.dt.dayofweek

# %%


# look at errors over time

for count_type, count_time in [(x,y) for x in count_types for y in count_times]:

    obs_plot = observation_df_filt[['area [hex]', 'poi_nuid', 'count_date', f'pct_residual_{count_type}_{count_time}']]
    obs_plot['res_bins'], bins = pd.cut(obs_plot[f'pct_residual_{count_type}_{count_time}'], bins=9, retbins=True)

    obs_plot.sort_values('count_date', inplace=True)
    obs_plot['zscored residual'] = obs_plot.groupby('poi_nuid')[f'pct_residual_{count_type}_{count_time}'].transform(stats.zscore)

    obs_plot = obs_plot[(obs_plot['count_date']>= pd.to_datetime('2024-06-01')) & (obs_plot['count_date']<= pd.to_datetime('2025-07-01'))]

    obs_plot['count_date_ord'] = obs_plot['count_date'].map(dt.datetime.toordinal)

    pop_result = stats.linregress(obs_plot['count_date_ord'], obs_plot['zscored residual'])

    fig, ax = plt.subplots()

    sns.lineplot(  
        data=obs_plot,
        x="count_date_ord",
        y="zscored residual",
        estimator='median',
        errorbar='pi',
        ax=ax
    )

    sns.regplot(data=obs_plot, x="count_date_ord", y="zscored residual", ax=ax, 
                line_kws={"label": "Regression Line"}, 
                scatter=False, ci=None)

    plt.xticks(rotation=45)
    plt.xlabel('date')
    plt.ylim((-5, 5))

    slope, intercept, r, p_value, sterr = stats.linregress(
        x=obs_plot['count_date_ord'], 
        y=obs_plot['zscored residual'],
        nan_policy='omit')

    xticklabels=[
        pd.to_datetime(x) for x in [
            # '2024-01', '2024-03','2024-05',
            '2024-07', '2024-09', '2024-11', 
            '2025-01', '2025-03', '2025-05', 
            '2025-07',]]

    ord_to_date = obs_plot.groupby('count_date_ord')[['count_date']].first().reset_index()
    xticks = ord_to_date[ord_to_date['count_date'].isin(xticklabels)]['count_date_ord'].tolist()
    xticklabels = [x.strftime('%Y-%m') for x in xticklabels]

    new_line = '\n'

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.text(
        0.08, 
        0.94, 
        f"slope: {365*slope:.2f} z/year{new_line}p-value: {p_value:.2f}",  
        ha="left", 
        va="center", 
        transform=ax.transAxes,
        )
    
    fig.suptitle(f'{count_type} {count_time}')

    fig.savefig(f'./figures/error_analysis/residual_vs_time_{count_type}_{count_time}.png', bbox_inches='tight')


# look at errors in space across London



# %%
# add size bin flag to the main dataframe

observation_df_filt.loc[observation_df_filt['area [hex]']<=0.5, 'size_flag'] = 'tiny'
observation_df_filt.loc[(observation_df_filt['area [hex]']>0.5)&(observation_df_filt['area [hex]']<=1), 'size_flag'] = 'small'
observation_df_filt.loc[(observation_df_filt['area [hex]']>1)&(observation_df_filt['area [hex]']<=2), 'size_flag'] = 'one-ish'
observation_df_filt.loc[(observation_df_filt['area [hex]']>2)&(observation_df_filt['area [hex]']<=5), 'size_flag'] = 'medium'
observation_df_filt.loc[(observation_df_filt['area [hex]']>5), 'size_flag'] = 'large'



# %%
# Harleyford Road example

hfrd_id = 190

df_harleyford = observation_df_filt[observation_df_filt['poi_nuid'] == 'highstreets_190']

hfrd_name = df_harleyford.poi_name.iloc[0]
hfrd_nuid = df_harleyford.poi_nuid.iloc[0]
hfrd_area = df_harleyford.area.iloc[0]
hfrd_area_hex = df_harleyford['area [hex]'].iloc[0]
# hfrd_area_hex = df_harleyford['area [hex]'].iloc[0]


hfrd_scale_factors = []
for count_type, count_time in [(x,y) for x in count_types for y in count_times]:
    hfrd_scale_factors.append(
        {
            'count_type': count_type,
            'count_time': count_time,
            'scale_factor_model': sigmoid(np.log(hfrd_area_hex), *model_params.loc[(model_params['count_type']==count_type) & (model_params['count_time']==count_time), ['L','k','x0']].iloc[0].tolist()),
            'scale_factor_true_avg': df_harleyford[f'scale_factor_true_{count_type}_{count_time}'].mean()
        }
    )

hfrd_scale_factors = pd.DataFrame(hfrd_scale_factors)

df_harleyford.drop(columns=[
    'poi_nuid', 'poi_id', 'poi_uid', 'poi_name', 
    'area', 'area_bin', 'area [hex]', 'poi_type', 'caz_inner_outer'], 
    inplace=True)


df_harleyford.reset_index(drop=True, inplace=True)
df_harleyford = df_harleyford.loc[:, ~df_harleyford.columns.str.contains('per_area')]
# df_harleyford = df_harleyford.loc[:, ~df_harleyford.columns.str.contains('model_scale_factor')]
df_harleyford = df_harleyford.loc[:, ~df_harleyford.columns.str.contains('overcount_ratio')]
df_harleyford = df_harleyford.loc[:, ~df_harleyford.columns.str.contains('avg_dwell_time')]

df_harleyford.rename(
    columns={f'{x}_{y}': f'duplicated_{x}_{y}' for x in count_types for y in count_times},
    inplace=True)

for count_type in count_types:
    df_harleyford.columns = df_harleyford.columns.str.replace(f'{count_type}s', count_type)

df_harleyford["id"] = df_harleyford.index

df_harleyford = pd.wide_to_long(
    df_harleyford,
    stubnames=[
        "total_unique", "duplicated", "pct_residual", 
        "predicted", "scale_factor_model", "scale_factor_true"],
    i="id",
    j="type_time",
    sep='_',
    suffix=r"\w+"
)

df_harleyford = df_harleyford.reset_index().drop(columns=['id'])

df_harleyford[['count_type', 'count_time']] = df_harleyford['type_time'].str.split('_', expand=True)
df_harleyford.drop(columns=['type_time'], inplace=True)


# # type x time grid of plots BT vs model
# g = sns.FacetGrid(df_harleyford, col='count_type', row='count_time', sharex=False, sharey=False)
# g.map(sns.scatterplot, 'total_unique', 'predicted')

# %%
sns.set_theme(font_scale=0.9)

g = sns.FacetGrid(
    data=df_harleyford.drop(columns=['count_date', 'size_flag']).groupby(['count_type', 'count_time', 'weekday']).mean().reset_index(),
    col='count_time',
    row='count_type',
    sharey=False, sharex=False
)
g.map(sns.scatterplot, 'duplicated','scale_factor_true')

# %%
g = sns.FacetGrid(
    data=df_harleyford.drop(columns=['weekday', 'size_flag']).groupby(['count_type', 'count_time', 'count_date']).mean().reset_index(),
    col='count_time',
    row='count_type',
    sharey=False, sharex=False
)
g.map(sns.scatterplot, 'duplicated','scale_factor_true')

# %%

examples = []
for size_bin  in ['tiny', 'one-ish', 'large']:
    example_ids = random.sample(observation_df_filt[observation_df_filt['size_flag']==size_bin]['poi_nuid'].unique().tolist(), 3)

    example_df = observation_df_filt.loc[observation_df_filt['poi_nuid'].isin(example_ids), ['poi_nuid', 'worker_day', 'total_unique_workers_day', 'scale_factor_true_worker_day', 'size_flag', 'weekday']]

    example_df.loc[:, 'example_number'], _ = pd.factorize(example_df['poi_nuid'])

    examples.append(example_df)

    print(example_ids)

example_df = pd.concat(examples)

example_df.loc[example_df['weekday'].isin([0,1,2,3,4]),'day_type'] = 'week'
example_df.loc[example_df['weekday'].isin([5,6]),'day_type'] = 'weekend'

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
# %%

g = sns.FacetGrid(
    data=example_df,
    row='size_flag',
    col='example_number',
    palette='rocket',
    sharey=False, sharex=False, 
)

g.map_dataframe(
    sns.scatterplot, 
    'worker_day',
    'total_unique_workers_day', 
    hue='day_type', 
    style='weekday',
    alpha=0.5 
)

# %%

df_plot = (df_harleyford[df_harleyford['count_type']=='resident']
           .drop(columns=['size_flag', 'weekday'])
           .groupby(['count_type', 'count_time', 'count_date'])
           .mean()
           .reset_index()
)

g = sns.FacetGrid(
    data=df_plot,
    col='count_time',
    row='count_type',
    sharey=False, sharex=False
)
g.map(sns.scatterplot, 'total_unique', 'duplicated')



# %%

sns.relplot(
    data=df_harleyford, 
    x='total_unique', 
    y='predicted',
    col='count_type',
    hue='count_time', 
    facet_kws={'sharey': False, 'sharex': False}
    )

# map Harleyford road

# comparison with some other areas



# %%
from sklearn.metrics import mean_absolute_percentage_error

# show that generally speaking the deduplicated count is a linear function of the duplicated count
# observation_df_filt[observation_df_filt['poi_nuid']=='']



# set up metrics

def evaluate_dedupe(
        data, 
        true_deduped_col='total_count', 
        deduped_col='model_count', 
        type_col='count_type',
        time_col='count_time',
        ):

    # MAPE across all data
    mape_all = mean_absolute_percentage_error(data[true_deduped_col], data[deduped_col])

    # MAPE for each combination of type x time
    mape_grouped = data.groupby([type_col, time_col])

    