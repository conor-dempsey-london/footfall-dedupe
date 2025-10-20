# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import boto3
import arviz as az
import pymc as pm
import seaborn as sns

# get email password - figure out with tfl email to talk to

sns.set_theme(style="ticks")

from hs_models.utils import (
    load_footfall_dedupe_data, 
    clean_sample_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
    plot_data_examples,
    plot_data,
    fit_line,
    plot_moderation_effect,
    posterior_prediction_plot,
    make_scalarMap,
    make_all_plots
)

observation_df, area_df = load_footfall_dedupe_data()

observation_df_filt, stats_df = clean_sample_footfall_dedupe_data(
    observation_df,
    area_df,
)

sample_data = get_sample_of_footfall_dedupe_data(
    observation_df_filt,
    n_sample_areas = 36,
    n_obs_per_area = 50
)

stats_df_sample = stats_df[stats_df['poi_nuid'].isin(sample_data['poi_nuid'].unique())]

area_id, areas = sample_data.poi_nuid.factorize()
coords = {"area_id": areas}

sample_data['area_id'] = area_id

m_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

RANDOM_SEED = 1456

# %%
count_types = ['resident', 'worker', 'visitor']
time_indicators = {'day', 'am', 'pm'}

partial_pooling={
    y: {x: None for x in time_indicators} for y in count_types
}

for count_type in count_types:
    for time_indicator in time_indicators:

        with pm.Model(coords=coords) as partial_pooling[count_type][time_indicator]:
            area_idx = pm.Data("area_idx", area_id, dims="obs_id")

            x = pm.Data(
                    "x", 
                    sample_data[f'{count_type}s_per_area_{time_indicator}'].values,
                    dims="obs_id",
                )
            m = pm.Data("m", sample_data['area'].values)

            # priors
            β0 = pm.Normal("β0", mu=0, sigma=100)

            mu_b = pm.Normal("mu_b", mu=0.0, sigma=10)
            sigma_b = pm.Exponential("sigma_b", 1)

            β1 = pm.Normal("β1", mu=0, sigma=10)

            β2 = pm.Normal("β2", mu=mu_b, sigma=sigma_b, dims="area_id")

            σ = pm.HalfCauchy("σ", 10000)

            mu = β0 + β1*x + β2[area_idx]*x*m

            # likelihood
            y = pm.Normal(
                "y", 
                mu=mu, 
                sigma=σ, 
                observed=sample_data[f'unique_{count_type}s_per_area_{time_indicator}'].values, 
                dims="obs_id",
                )

pm.model_to_graphviz(partial_pooling[count_type][time_indicator])
# %%

partial_traces={
    y: {x: None for x in time_indicators} for y in count_types
}

for count_type in count_types:
    for time_indicator in time_indicators:
        with partial_pooling[count_type][time_indicator]:
            partial_traces[count_type][time_indicator] = pm.sample(
                random_seed=RANDOM_SEED, 
                target_accept=0.99,
            )

# %%

cols_select = [
    'workers_per_area_day', 'workers_per_area_pm', 'workers_per_area_am',
    'visitors_per_area_day', 'visitors_per_area_pm', 'visitors_per_area_am',
    'residents_per_area_day', 'residents_per_area_pm', 'residents_per_area_am',
]

# Compute the correlation matrix
corr = sample_data[cols_select].corr()

# Generate a mask for the upper triangle
mask = (~np.triu(np.ones_like(corr, dtype=bool))).transpose()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.color_palette("flare", as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr, 
    mask=mask, 
    cmap=cmap,
    square=True, 
    linewidths=.5, 
    cbar_kws={"shrink": .5},
)

f.savefig('./figures/count_correlation_matrix_by_type_and_time.png')

# %%

beta_2_dict = {}

for count_type in count_types:
    for time_indicator in time_indicators:
        beta_2_dict[f'{count_type}_{time_indicator}'] = np.mean(np.mean(partial_traces[count_type][time_indicator].posterior['β2'],0),0).values

beta_2_df = pd.DataFrame.from_dict(beta_2_dict)

# Compute the correlation matrix
corr = beta_2_df.corr()

# Generate a mask for the upper triangle
mask = (~np.triu(np.ones_like(corr, dtype=bool))).transpose()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.color_palette("flare", as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr, 
    mask=mask, 
    cmap=cmap,
    square=True, 
    linewidths=.5, 
    cbar_kws={"shrink": .5},
)

f.savefig('./figures/beta_2s_correlation_matrix.png')

# %%

plt.scatter(beta_2_df['resident_am'], beta_2_df['visitor_pm'])

# %%

beta_2_df.scatter_matrix()

# %%

# look at temporal correlation between types (workers etc) 

# look at average correlation 

# investigate the visitor beta_2s - how does the distribution look?

# do all of these models fit well? Specifically the visitor beta_2s

# we could deploy the 3 models (worker, visitor, resident)

# one set of data in, one set of data out (eg if someone supplies just AM data they should get just AM deduped data out)

# 