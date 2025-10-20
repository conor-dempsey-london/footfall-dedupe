# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import boto3
import arviz as az
import pymc as pm
import seaborn as sns

sns.set_theme(style="ticks")

from hs_models.utils import (
    load_footfall_dedupe_data, 
    clean_sample_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
    plot_data,
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

area_id, areas = sample_data.poi_uid.factorize()
coords = {"area_id": areas}

m_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

RANDOM_SEED = 1456

# %%
# Plot histograms of predictors and outcome

count_types = ['resident', 'worker', 'visitor']

fig, ax = plt.subplots(3, 3, figsize=(16, 12), sharex=False, sharey=False)

m_data = sample_data['area']
# Make a scalar color map for this dataset (Just for plotting, nothing to do with inference)
scalarMap = make_scalarMap(m_data)
x_data={}
y_data={}

for count_idx, count_type in enumerate(count_types):

    x_data[count_type] = sample_data[f'{count_type}s_per_area']

    y_data[count_type] = sample_data[f'unique_{count_type}s_per_area']

    ax[count_idx][0].hist(x_data[count_type], alpha=0.5)
    ax[count_idx][0].set(xlabel=f"counted {count_type}s / km2")

    ax[count_idx][1].hist(y_data[count_type], alpha=0.5)
    ax[count_idx][1].set(xlabel=f"unique {count_type}s / km2")


    ax[count_idx][2].hist(m_data, alpha=0.5)
    ax[count_idx][2].set(xlabel="area (km2)")


# %%

with pm.Model() as pooled_model:
    x = pm.Data("x", x_data)
    m = pm.Data("m", m_data)

    # priors
    β0 = pm.Normal("β0", mu=0, sigma=100)
    β1 = pm.Normal("β1", mu=0, sigma=10)
    β2 = pm.Normal("β2", mu=0, sigma=10)
    β3 = pm.Normal("β3", mu=0, sigma=10)
    σ = pm.HalfCauchy("σ", 10000)

    # likelihood
    y = pm.Normal("y", mu=β0 + (β1 * x) + (β2 * x * m) + (β3 * m), sigma=σ, observed=y_data)

pm.model_to_graphviz(pooled_model)

# %%

with pm.Model(coords=coords) as unpooled_model:
    area_idx = pm.Data("area_idx", area_id, dims="obs_id")

    x = pm.Data("x", x_data.values, dims="obs_id")
    m = pm.Data("m", m_data.values)

    # priors
    β0 = pm.Normal("β0", mu=0, sigma=100)

    β1 = pm.Normal("β1", mu=0, sigma=10)

    β2 = pm.Normal("β2", mu=0, sigma=10, dims="area_id")

    σ = pm.HalfCauchy("σ", 10000)

    mu = β0 + β1*x + β2[area_idx]*x*m

    # likelihood
    y = pm.Normal("y", mu=mu, sigma=σ, observed=y_data.values, dims="obs_id")

pm.model_to_graphviz(unpooled_model)

# %%
partial_pooling={x: None for x in count_types}
for count_type in count_types:
    with pm.Model(coords=coords) as partial_pooling[count_type]:
        area_idx = pm.Data("area_idx", area_id, dims="obs_id")

        x = pm.Data("x", x_data[count_type].values, dims="obs_id")
        m = pm.Data("m", m_data.values)

        # priors
        β0 = pm.Normal("β0", mu=0, sigma=100)

        mu_b = pm.Normal("mu_b", mu=0.0, sigma=10)
        sigma_b = pm.Exponential("sigma_b", 1)

        β1 = pm.Normal("β1", mu=0, sigma=10)

        β2 = pm.Normal("β2", mu=mu_b, sigma=sigma_b, dims="area_id")

        σ = pm.HalfCauchy("σ", 10000)

        mu = β0 + β1*x + β2[area_idx]*x*m

        # likelihood
        y = pm.Normal("y", mu=mu, sigma=σ, observed=y_data[count_type].values, dims="obs_id")

pm.model_to_graphviz(partial_pooling[count_type])

# %%
with pm.Model(coords=coords) as partial_pooling_variable_intercept:
    area_idx = pm.Data("area_idx", area_id, dims="obs_id")

    x = pm.Data("x", x_data.values, dims="obs_id")
    m = pm.Data("m", m_data.values)

    # priors
    mu_b = pm.Normal("mu_b", mu=0.0, sigma=10)
    sigma_b = pm.Exponential("sigma_b", 1)

    mu_a = pm.Normal("mu_a", mu=0.0, sigma=1000)
    sigma_a = pm.Exponential("sigma_a", 1000)

    β0 = pm.Normal("β0", mu=mu_a, sigma=sigma_a, dims="area_id")

    β1 = pm.Normal("β1", mu=0, sigma=10)

    β2 = pm.Normal("β2", mu=mu_b, sigma=sigma_b, dims="area_id")

    σ = pm.HalfCauchy("σ", 10000)

    mu = β0[area_idx] + β1*x + β2[area_idx]*x*m

    # likelihood
    y = pm.Normal("y", mu=mu, sigma=σ, observed=y_data.values, dims="obs_id")

pm.model_to_graphviz(partial_pooling_variable_intercept)

# %%

with pooled_model:
    pooled_trace = pm.sample(random_seed=RANDOM_SEED, target_accept=0.95)

#%%
with unpooled_model:
    unpooled_trace = pm.sample(random_seed=RANDOM_SEED, target_accept=0.99)

# %%
partial_traces = {}
for count_type in count_types:
    with partial_pooling[count_type]:
        partial_traces[count_type] = pm.sample(random_seed=RANDOM_SEED, target_accept=0.99)

# %%
with partial_pooling_variable_intercept:
    partial_varint_trace = pm.sample(random_seed=RANDOM_SEED, target_accept=0.99)

# %%
with pooled_model:
    pm.compute_log_likelihood(pooled_trace)

with unpooled_model:
    pm.compute_log_likelihood(unpooled_trace)

with partial_pooling:
    pm.compute_log_likelihood(partial_trace)

with partial_pooling_variable_intercept:
    pm.compute_log_likelihood(partial_varint_trace)

# %%

df_comp_loo = az.compare(
    {
        "partial pooling": partial_trace, 
        "unpooled": unpooled_trace,
        "partial pooling varint": partial_varint_trace})
df_comp_loo

# %%
az.plot_compare(df_comp_loo, insample_dev=False)

# %%

make_all_plots(pooled_trace, scalarMap)

# %%
beta_2={}
for count_type in count_types:
    beta_2[count_type] = np.mean(np.mean(partial_traces[count_type].posterior['β2'],0),0).values

beta_2_df = pd.DataFrame.from_dict(beta_2)

sns.pairplot(beta_2_df)

# %%
# trace_to_plot = partial_varint_trace
count_type='visitor'

trace_to_plot = partial_traces[count_type]
# trace_to_plot = unpooled_trace

fig, ax = plt.subplots(figsize=(10, 6))

b0 = np.mean(trace_to_plot.posterior['β0']).item()
b1 = np.mean(trace_to_plot.posterior['β1']).item()
b2 = np.mean(np.mean(trace_to_plot.posterior['β2'], 0), 0).values

area_vals = trace_to_plot.constant_data.m.values[area_id]
x_vals = trace_to_plot.constant_data.x.values
m_vals = trace_to_plot.constant_data.m.values

mu_trace = b0 + b1 * x_vals + np.multiply(np.multiply(b2[trace_to_plot.constant_data.area_idx], x_vals), m_vals)

n_areas = len(np.unique(trace_to_plot.constant_data.area_idx.values))
obs_per_area = int(len(trace_to_plot.constant_data.area_idx.values) / n_areas)

plot_data(x_vals, m_vals, trace_to_plot.observed_data.y.values, scalarMap, ax=ax)

for i in range(n_areas):
    ax.plot(x_vals[i*obs_per_area:(i+1)*obs_per_area - 1], mu_trace[i*obs_per_area:(i+1)*obs_per_area - 1], color='black', alpha=0.3)

ax.set_xlabel('counted residents per km2')
ax.set_ylabel('unique residents per km2')

ax.set_title("Data and posterior prediction")

# %%
beta_0 = np.round(trace_to_plot.posterior['β0'].mean().item())
beta_1 = trace_to_plot.posterior['β1'].mean().item()
beta_2 = trace_to_plot.posterior['β2'].mean().item()

print(f'β0: {beta_0}')
print(f'β1 * <x> : {(beta_1*x_data.mean()).round()}')
print(f'β2 * <x> * <m>: {(beta_2*x_data.mean()*m_data.mean()).round()}')

# %%

az.plot_trace(trace_to_plot, figsize=(16, 22))

# %%
areas_unique = np.unique(trace_to_plot.constant_data.m)

beta_2_std = np.std(np.median(trace_to_plot.posterior['β2'], 0),0)

beta_2_means = np.median(np.median(trace_to_plot.posterior['β2'], 0), 0)

fig, ax=plt.subplots()

yplot = beta_1 + trace_to_plot.posterior.mu_b.median().item()*areas_unique
err_plot = trace_to_plot.posterior.sigma_b.median().item()*areas_unique

ax.plot(areas_unique, yplot, color='#000000') 
ax.fill_between(areas_unique, yplot-err_plot, yplot+err_plot, alpha=0.5, edgecolor='#808080', facecolor='#808080')

ax.scatter(areas_unique, beta_1 + np.multiply(areas_unique, beta_2_means))
ax.scatter(stats_df_sample['area'], stats_df_sample['slope'])

ax.set_ylim(-0.02, 0.3)
ax.set(
    xlabel="$area$ (km2)",
    ylabel=r"$\beta_1 + \beta_2 \cdot area$",
)

# %%

az.plot_posterior(trace_to_plot, var_names=["β0", "β1", "β3"], figsize=(14, 4))

az.plot_forest(trace_to_plot, var_names="β2")


# %%
# Plot example areas with fits from different models
n_plot_x = 4
n_plot_y = 3
n_plot = n_plot_x * n_plot_y

fig, ax=plt.subplots(n_plot_x, n_plot_y, figsize=(20, 20))
ax = ax.flatten()

area_indices = trace_to_plot.constant_data.area_idx.values
area_idx_unique = np.unique(area_indices)
areas_plot = np.random.choice(area_idx_unique, n_plot)

to_plot_mask = np.isin(area_indices, areas_plot)

b2 = np.mean(np.mean(trace_to_plot.posterior['β2'], 0), 0).values

b0_pooled = np.mean(pooled_trace.posterior['β0']).item()
b1_pooled = np.mean(pooled_trace.posterior['β1']).item()
b2_pooled = np.mean(pooled_trace.posterior['β2']).item()

x_plot = trace_to_plot.constant_data.x[to_plot_mask]
m_plot = trace_to_plot.constant_data.m[to_plot_mask]
area_idx_plot = trace_to_plot.constant_data.area_idx[to_plot_mask]
y_plot = trace_to_plot.observed_data.y[to_plot_mask]

mu_plot = b0 + b1 * x_plot.values + np.multiply(np.multiply(b2[area_idx_plot], x_plot.values), m_plot.values)

mu_pooled = b0_pooled + b1_pooled * x_plot.values + b2_pooled * np.multiply(x_plot.values, m_plot.values)

n_areas = len(np.unique(area_idx_plot))
obs_per_area = int(len(area_idx_plot) / n_areas)

# plot_data(x_plot, m_plot, y_plot, scalarMap, ax=ax)

for i in range(n_plot):
    ax[i].scatter(x_plot[i*obs_per_area:(i+1)*obs_per_area - 1], y_plot[i*obs_per_area:(i+1)*obs_per_area - 1], s=76)

    ax[i].plot(x_plot[i*obs_per_area:(i+1)*obs_per_area - 1], mu_plot[i*obs_per_area:(i+1)*obs_per_area - 1], color='black', alpha=0.7, linewidth=4)

    ax[i].plot(x_plot[i*obs_per_area:(i+1)*obs_per_area - 1], mu_pooled[i*obs_per_area:(i+1)*obs_per_area - 1], color='orange', alpha=0.7, linewidth=4)

    ax[i].set_xlabel('counted residents per km2')
    ax[i].set_ylabel('unique residents per km2')

    # ax[i].set_title("Data and posterior prediction")


# %%

# example of making predictions for a new area with 
areas_new = [0.6, 0.6]

x_new, m_new, area_idx_new = [], [], []
for idx, area_new in enumerate(areas_new):
    x_new_next = np.sort(np.random.normal(170000, 50000, (200,)))
    m_new_next = np.full(x_new_next.shape, area_new)
    area_idx_new_next = np.full(x_new_next.shape, idx)

    x_new.append(x_new_next)
    m_new.append(m_new_next)
    area_idx_new.append(area_idx_new_next)

x_new = np.concatenate(tuple(x_new))
m_new = np.concatenate(tuple(m_new))
area_idx_new = np.concatenate(tuple(area_idx_new))

with partial_pooling:
    pm.set_data(
        {
            "m": m_new, 
            "x": x_new,
            "area_idx": area_idx_new
        },
        coords = {"area_id": ["HIGH00NEW", "HIGH00NEW2"]} 
    )
    predictive_results = pm.sample_posterior_predictive(
        partial_trace, 
        predictions=True,
    )


# %%
fig, ax = plt.subplots(figsize=(6, 5))

y_traces = np.reshape(predictive_results.predictions.y.values, (4000, 400))

y_traces_1 = y_traces[:100:, :200]
y_traces_2 = y_traces[:100:, 201:]

ax.plot(x_new[:200], np.transpose(y_traces_1), color='black', alpha=0.01)
ax.plot(x_new[201:], np.transpose(y_traces_2), color='black', alpha=0.01)
ax.plot(
    x_new[:200], 
    np.mean(y_traces_1,0)
)
ax.plot(
    x_new[201:], 
    np.mean(y_traces_2,0)
)
ax.set_xlabel('counted residents per km2')
ax.set_ylabel('unique residents per km2')

# %%
rng = np.random.default_rng(RANDOM_SEED)

# %%

with partial_pooling:
    pm.sample_posterior_predictive(partial_trace, extend_inferencedata=True, random_seed=rng)


# %%
az.plot_ppc(partial_trace, num_pp_samples=100)



# %%

# make posterior predictions
with partial_pooling:
    pm.sample_posterior_predictive(
        partial_trace, 
        extend_inferencedata=True,
    )

# Make finer-grained prior and posterior predictive checks and comparisons for the pooled and partially-pooled models

# %%
n_sample = 12
sample_areas = np.random.choice(np.unique(area_id), n_sample, replace=False)

area_idx_sample = np.isin(partial_trace.constant_data.area_idx, sample_areas)

x_sample = partial_trace.constant_data.x[area_idx_sample]
m_sample = partial_trace.constant_data.x[area_idx_sample]
y_sample = partial_trace.observed_data.y[area_idx_sample]

y_post_pred_sample = np.mean(partial_trace.posterior_predictive.y[:, :, area_idx_sample],1).transpose()

β0_map = np.mean(partial_trace.posterior.β0).item()
mu_b_map = np.mean(partial_trace.posterior.mu_b).item()
β1_map = np.mean(partial_trace.posterior.β1).item()
β2_map_pooled = np.mean(pooled_trace.posterior.β1).item()

β2s_map = np.mean(
        np.mean(partial_trace.posterior.β2,0),0
    ).values[sample_areas]

fig, axes = plt.subplots(4,3,figsize=(15,15))
axes = axes.flatten()

obs_per_area = int(len(x_sample) / n_sample)

for idx, area_idx in enumerate(sample_areas):
    i = idx*obs_per_area
    j = (idx+1)*obs_per_area

    x_current = x_sample[i:j]
    y_current = y_sample[i:j]
    m_current = m_sample[i:j]
    y_post_pred_current = y_post_pred_sample[i:j, :]

    x_sort_order = np.argsort(x_current).values

    axes[idx].scatter(x_current[x_sort_order], y_current[x_sort_order])
    axes[idx].plot(x_current[x_sort_order], y_post_pred_current[x_sort_order, :], color='black', alpha=0.3)

    y_mean_partial = β0_map + β1_map*x_current + β2s_map[idx] * np.mean(m_current) * x_current

    axes[idx].plot(x_current[x_sort_order], y_mean_partial[x_sort_order])




# %%

# Deadline: model repo tidied up and all these points done by the 25th!!

# TODO: (PRIORITY THREE) clean up code and upload to github
# TODO: (PRIORITY FOUR) look at distance from center of London
# TODO: (PRIORITY FIVE) visitor dwell time and loyalty percentage columns - these might also be predictive in a deduping model? Let's have a look at this
# TODO: (PRIORITY SIX) leave a sample of data out and then look at the success of actual deduplicating (maybe look at HS and test on Town Centers)
# TODO: look at type of area (HS, BIG, Towncenter etc) - does this matter in some way independently of area?
# TODO: look at the saturation of slopes for larger areas: can the model capture this?
# TODO: implement robust regression to decrease the effect of outliers?
# TODO: does the geometry of the bespoke area affect deduping in any way? 
# TODO: include time explicitly?

# Look first at:
# 1. Clean up the code
# 2. Have a set of standard plots we are making for all models
# 3. Have a set of plots that are specific to certain models
# 4. Standardise model comparison methods


# %%

# TODO: (PRIORITY ONE) look at workers and visitors as well - how similar are beta_2 values for these?


