# %% setup and data loading
import arviz as az
import preliz as pz
import pymc as pm
import numpy as np
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt

from hs_models.utils import (
    load_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
)
from hs_models.models import (
    LinearPooledNoInteraction,
    LinearPartPoolNoInteraction,
    AreaCountInteraction1DPartPoolPositive,
)

from dotenv import load_dotenv

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

sample_data = get_sample_of_footfall_dedupe_data(
    observation_df_filt,
    n_sample_areas = 36,
    n_obs_per_area = 50
)

count_type ='worker'
count_time = 'day'

X = {}
y = {}
models = {}

count_type='worker'

X = sample_data[
    [
        'poi_nuid', 
        f'{count_type}s_per_area_{count_time}', 
        'area',
    ]
    ].rename(columns={
        f'{count_type}s_per_area_{count_time}': 'total_counts',
        'area': 'area_size',
        'poi_nuid': 'area_id',
    },
)

y = sample_data[f'total_unique_{count_type}s_day']

# %% 0. data plotting util
def plot_sample_data(X, y):
    X_all = X.copy()
    X_all['deduped counts'] = y.copy()
    X_all.rename(columns={'total_counts': 'total counts'}, inplace=True)

    n_x=3 
    n_y = 3
    n=n_x * n_y
    n_sample = 1 

    b1 = -0.005
    b2 = 0.185

    area_bins = np.linspace(X_all['area_size'].min(), X_all['area_size'].max(), n+1)

    X_all['area_bin'] = pd.cut(X_all['area_size'], area_bins)

    sample_areas = X_all.groupby('area_bin').apply(lambda x: x.sample(1), include_groups=False)['area_id']

    X_plot = X_all[X_all['area_id'].isin(sample_areas)].sort_values('area_bin')

    def scatter_w_fit(x, y, z, **kwargs):
        sns.scatterplot(x=x, y=y, **kwargs)
        xmin, xmax = (
            x.min(), 
            x.max()
        )
        ymin, ymax = (
            b1*x.min() + b2*x.min()*z.mean(), 
            b1*x.max() + b2*x.max()*z.mean(),
        )
        plt.plot((xmin, xmax), (ymin, ymax), '-k')

    g = sns.FacetGrid(X_plot, col='area_id', hue='area_bin', palette='flare', col_wrap=3)
    g.map(scatter_w_fit, 'total counts', 'deduped counts', 'area_size')

def plot_data_prior_posterior(model):

    df = pd.DataFrame(
        {
            'total_counts': model.idata.fit_data.total_counts,
            'area': model.idata.fit_data.area_size,
            'y obs': model.idata.fit_data.y,
            'y prior': model.idata.prior_predictive.y.mean(dim=('chain', 'draw')),
            'y posterior': model.idata.posterior_predictive.y.mean(dim=('chain', 'draw')),
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    sns.scatterplot(df, x='total_counts', y='y obs', hue='area', ax=axes[0])
    sns.scatterplot(df, x='total_counts', y='y prior', hue='area', ax=axes[1])
    sns.scatterplot(df, x='total_counts', y='y posterior', hue='area', ax=axes[2])

    return fig, axes

# %% 1. Plot some data
plot_sample_data(X, y)

# %% 2. Baseline model

model_config = { 
    "β1_mu_prior": 0.02,
    "β1_sigma_prior": 0.03,
    "σ_prior": 600.0
}

baseline_model = LinearPooledNoInteraction(model_config)

# %% 3. Prior predictive checks
prior_predictive_samples = baseline_model.sample_prior_predictive(X)
plot_sample_data(X, prior_predictive_samples.y.mean(dim=('sample')))

# %% 4. Fit model

baseline_model.fit(X, y)

# %% 5. Posterior checks

az.plot_trace(baseline_model.idata, figsize=(20,10))

az.summary(baseline_model.idata)

plot_data_prior_posterior(baseline_model)

# %% Next model - B1 partial pooling

partpool_no_b2_model = LinearPartPoolNoInteraction()

partpool_no_b2_model.sample_prior_predictive(X)
plot_sample_data(X, partpool_no_b2_model.prior_predictive.y.mean(dim=('sample')))
partpool_no_b2_model.fit(X, y)
az.plot_trace(partpool_no_b2_model.idata, figsize=(20,10))
az.summary(partpool_no_b2_model.idata)
plot_data_prior_posterior(partpool_no_b2_model)


# %%
