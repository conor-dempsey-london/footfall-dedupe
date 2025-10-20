# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import boto3
import bambi as bmb
import arviz as az

from hs_models.utils import (
    load_sample_footfall_dedupe_data, 
    clean_sample_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
    plot_data,
    fit_line
)

observation_df, area_df = load_sample_footfall_dedupe_data()

observation_df_filt, stats_df = clean_sample_footfall_dedupe_data(
    observation_df,
    area_df,
)

sample_data = get_sample_of_footfall_dedupe_data(
    observation_df_filt,
    n_sample_areas = 12,
    n_obs_per_area = 100
)

# %%
# plot the area against the intercept
stats_df.plot.scatter('area', 'intercept')

# plot the area against the slope
stats_df.plot.scatter('area', 'slope')

# %%
print(f'Average slope: {stats_df['slope'].mean()}')
print(f'Average intercept: {stats_df['intercept'].mean()}')

m_slope_vs_area, b_slope_vs_area = fit_line(stats_df['area'], stats_df['slope'])

print(f'Area vs slope slope: {m_slope_vs_area}')

# %%
# look at the distribution of the slopes
stats_df[['slope', 'intercept']].plot.hist(column='slope', xlabel='slope', bins=25, alpha=0.6)

# %%
# look at the distribution of the slopes
stats_df[['slope', 'intercept']].plot.hist(column='intercept', xlabel='intercept', bins=25, alpha=0.6)


# %%
axes = plot_data(sample_data)

# %%

simple_pooled_model = bmb.Model(
    "unique_residents_per_area ~ 1 + (residents_per_area:area)", 
    sample_data, 
    categorical="poi_uid",
)

plt.style.use("arviz-darkgrid")
SEED = 1234

results = simple_pooled_model.fit(random_seed=SEED)

# %%

az.summary(results, var_names=[
    "Intercept", 
    "residents_per_area:area"], kind="stats")

# %%
az.plot_trace(results, var_names=[
    "Intercept", 
    "residents_per_area:area"
])

# %%

# Obtain the posterior of the mean
simple_pooled_model.predict(results)

axes = plot_data(sample_data)

# Take the posterior of the mean reaction time
unique_residents_mean = az.extract(results)["mu"].values

for poi_uid, ax in zip(sample_data["poi_uid"].unique(), axes.ravel()):

    idx = sample_data.index[sample_data["poi_uid"]== poi_uid].tolist()
    days = sample_data.loc[idx, "residents_per_area"].values
    
    # Plot highest density interval / credibility interval
    az.plot_hdi(days, unique_residents_mean[idx].T[np.newaxis], color="C0", ax=ax)
    
    # Plot mean regression line
    ax.plot(days, unique_residents_mean[idx].mean(axis=1), color="C0")


# %%
posterior = az.extract_dataset(results)

intercept_common = posterior['Intercept']
slope_common = posterior['residents_per_area:area']

# %%

simple_pooled_model.build()
simple_pooled_model.graph()

# %%

fig, ax = plt.subplots(figsize=(7, 3), dpi=120)
bmb.interpret.plot_predictions(
    simple_pooled_model, 
    results, "residents_per_area", pps=True, ax=ax)
    
# %%
