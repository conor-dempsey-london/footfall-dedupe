import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import boto3
import arviz as az
from typing import Tuple, Dict
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import xarray as xr


def load_footfall_dedupe_data(
    bucket, file_name, area_file
) -> (pd.DataFrame, pd.DataFrame):

    # Load the data
    s3 = boto3.client('s3') 
    obj_data = s3.get_object(Bucket= bucket, Key= file_name) 
    obj_area = s3.get_object(Bucket= bucket, Key= area_file) 

    # get object and file (key) from bucket
    observation_df = pd.read_csv(
        obj_data['Body'], 
        parse_dates=['count_date'],
        low_memory=False) 

    observation_df.rename(
        columns={'total_unique_domestic_visitors': 'total_unique_visitors'},
        inplace=True,
    )

    # get object and file (key) from bucket
    area_df = pd.read_csv(obj_area['Body']) 

    # Keep only the all-day numbers
    # day_mask = observation_df['time_indicator'] == 'DAY'
    # observation_df_filt = observation_df[day_mask]

    day_df = (
        observation_df[observation_df['time_indicator'] == 'DAY']
        .dropna(subset=['poi_uid', 'count_date'])
        .drop(columns='time_indicator')
        .rename(columns={
            'visitor': 'visitor_day', 
            'worker': 'worker_day', 
            'resident': 'resident_day', 'total_unique_workers': 'total_unique_workers_day', 'total_unique_residents': 'total_unique_residents_day',
            'total_unique_visitors': 'total_unique_visitors_day',
            'avg_dwell_time': 'avg_dwell_time_day'})
    )

    am_df = (
        observation_df[observation_df['time_indicator'] == 'AM']
        .dropna(subset=['poi_uid', 'count_date'])
        .drop(columns='time_indicator')
        .rename(columns={
            'visitor': 'visitor_am', 
            'worker': 'worker_am', 
            'resident': 'resident_am', 
            'total_unique_workers': 'total_unique_workers_am', 'total_unique_residents': 'total_unique_residents_am' ,
            'total_unique_visitors': 'total_unique_visitors_am','avg_dwell_time': 'avg_dwell_time_am'
            })
    )

    pm_df = (
        observation_df[observation_df['time_indicator'] == 'PM']
        .dropna(subset=['poi_uid', 'count_date'])
        .drop(columns='time_indicator')
        .rename(columns={
            'visitor': 'visitor_pm', 
            'worker': 'worker_pm', 
            'resident': 'resident_pm', 
            'total_unique_workers': 'total_unique_workers_pm', 'total_unique_residents': 'total_unique_residents_pm',
            'total_unique_visitors': 'total_unique_visitors_pm','avg_dwell_time': 'avg_dwell_time_pm'
            })
    )

    observation_df=day_df.merge(am_df, on=['poi_uid', 'count_date', 'poi_name', 'poi_id', 'poi_type', 'caz_inner_outer'], how='inner', suffixes=('_day', '_am'))

    observation_df=observation_df.merge(pm_df, on=['poi_uid', 'count_date', 'poi_name', 'poi_id', 'poi_type', 'caz_inner_outer'], how='inner', suffixes=('_day', '_pm'))

    observation_df['poi_nuid']=observation_df['poi_type'] + '_' + observation_df['poi_id'].astype(str)

    observation_df = observation_df[['poi_nuid'] + list(observation_df.columns[:-1])]

    area_df['poi_nuid'] = area_df['poi_type'] + '_' + area_df['poi_id'].astype(str)

    area_df = area_df[['poi_nuid'] + list(area_df.columns[:-1])]


    # take the mean area value for each area (in case there's noise?)
    # areas are measured in meters squared, rescale to use kilometers squared as this give more manageable numbers
    area_df = area_df.groupby('poi_nuid')['area'].mean().reset_index() 

    area_df['area'] = area_df['area'] / 1e6

    # add area data to main df
    observation_df = observation_df.merge(area_df[['poi_nuid', 'area']], on='poi_nuid', how='left')

    observation_df = observation_df[~observation_df['area'].isna()]

    # for now we remove the very largest areas which show less predictable behaviour
    observation_df = observation_df[observation_df['area'] < 0.4]

    # create area bins so we can take a stratified sample across the range of area sizes
    observation_df['area_bin'] = pd.qcut(observation_df['area'], q=6)

    for count_type in ['worker', 'resident', 'visitor']:

        for time_indicator in ['day', 'am', 'pm']:
                
            # for now we remove any areas that have zero unique residents, workers, or visitors
            observation_df = observation_df[observation_df[f'total_unique_{count_type}s_{time_indicator}'] > 0]

            # we will model counts per unit area as these have a more manageable scale
            observation_df[f'{count_type}s_per_area_{time_indicator}'] = (
                observation_df[f'{count_type}_{time_indicator}'] / observation_df['area']
            )

            observation_df[f'unique_{count_type}s_per_area_{time_indicator}'] = (
                observation_df[f'total_unique_{count_type}s_{time_indicator}'] / observation_df['area']
            ) 

            # compute best fit lines between observed and de-duped  counts for all areas
            res = (
                observation_df
                .groupby('poi_nuid')
                .apply(
                    lambda x: fit_line(
                        x[f'{count_type}s_per_area_{time_indicator}'], 
                        x[f'unique_{count_type}s_per_area_{time_indicator}']), 
                        include_groups=False,
                    )
            )

            stats_df = pd.DataFrame(res.tolist(), index=res.index, columns=[f'slope_{count_type}', f'intercept_{count_type}']).reset_index()

    return observation_df, stats_df


def get_sample_of_footfall_dedupe_data(
    observation_df_filt,
    n_sample_areas = 12,
    n_obs_per_area = 100
):
    sample_areas = (
        observation_df_filt
        .groupby('area_bin', observed=False)['poi_uid']
        .unique()
        .apply(lambda x: np.random.choice(
            x, 
            size=n_sample_areas, 
            replace=False,
            )).explode()
    )

    sample_data = (
        observation_df_filt[observation_df_filt['poi_uid']
        .isin(sample_areas)]
        .sort_values('count_date', ascending=False)
        .groupby('poi_uid')
        .apply(lambda x: x.sample(n_obs_per_area, replace=True), include_groups=False)
        .drop(columns=['poi_id', 'area_bin'])
        .reset_index()
    )

    sample_data.dropna(inplace=True)

    sample_data['area_ids'] = sample_data.groupby('poi_uid').ngroup()

    return sample_data

# fit a line to each di vs oi relationship
def fit_line(x, y):
    # Calculate the means of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the slope (m) using the least squares method
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    if denominator == 0:
        m = np.nan
    else:
        m = numerator / denominator

    # Calculate the y-intercept (b)
    b = y_mean - m * x_mean

    return m, b

def plot_data_examples(data, nx=6, ny=6):

    fig, axes = plt.subplots(
        nx, 
        ny, 
        figsize=(32, 19), 
        sharey=False, 
        sharex=False, 
        dpi=300, 
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.075, 
        right=0.975, 
        bottom=0.075, 
        top=0.925, 
        wspace=0.2,
    )

    axes_flat = axes.ravel()
    m, b, a = [], [], []

    for i, area_id in enumerate(data["poi_uid"].unique()):
        
        idx = data.index[data["poi_uid"] == area_id].tolist()
        resident = data.loc[idx, "residents_per_area"].values
        unique_residents = data.loc[idx, "unique_residents_per_area"].values

        if i < nx*ny:
            ax = axes_flat[i]
            # Plot observed data points
            ax.scatter(resident, unique_residents, color="C0", ec="black", alpha=0.7)

            # Add a title
            ax.set_title(f"area_id: {area_id}", fontsize=12)
    
    fig.text(0.5, 0.02, "residents", fontsize=14)
    fig.text(0.01, 0.5, "unique residents", rotation=90, fontsize=14, va="center")

    return axes

def center_data(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Center data by subtracting mean and dividing by standard deviation.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data to be centered
        
    Returns:
    --------
    Tuple[np.ndarray, Dict]
        centered_data: Centered and scaled data (mean=0, std=1)
        scaler_info: Dictionary with scaling parameters for uncentering
    """
    mean = np.mean(data)
    std = np.std(data)
    
    # Avoid division by zero
    if std == 0:
        std = 1.0
    
    centered_data = (data - mean) / std
    
    scaler_info = {
        'mean': mean,
        'std': std,
        'original_shape': data.shape,
        'original_dtype': data.dtype
    }
    
    return centered_data, scaler_info

def uncenter_data(
        centered_data: np.ndarray, 
        scaler_info: Dict,
    ) -> np.ndarray:
    """
    Reverse the centering transformation to return to original scale.
    
    Parameters:
    -----------
    centered_data : np.ndarray
        Centered data (mean=0, std=1)
    scaler_info : Dict
        Dictionary containing scaling parameters from center_data()
        
    Returns:
    --------
    np.ndarray
        Data in original scale
    """
    return centered_data * scaler_info['std'] + scaler_info['mean']


# some helper plotting functions

def make_scalarMap(m):
    """Create a Matplotlib `ScalarMappable` so we can use a consistent colormap across both data points and posterior predictive lines. We can use `scalarMap.cmap` to use as a colormap, and `scalarMap.to_rgba(moderator_value)` to grab a colour for a given moderator value."""
    return ScalarMappable(norm=Normalize(vmin=np.min(m), vmax=np.max(m)), cmap="viridis")


def plot_data(x, moderator, y, scalarMap, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()

    h = ax.scatter(x, y, c=moderator, cmap=scalarMap.cmap)
    ax.set(xlabel="x", ylabel="y")

    # colourbar for moderator
    cbar = fig.colorbar(h)
    cbar.ax.set_ylabel("area")
    return ax


def posterior_prediction_plot(result, x, moderator, m_quantiles, scalarMap, ax=None):
    """Plot posterior predicted `y`"""
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    post = az.extract(result)
    xi = xr.DataArray(np.linspace(np.min(x), np.max(x), 20), dims=["x_plot"])
    m_levels = result.constant_data["m"].quantile(m_quantiles).rename({"quantile": "m_level"})

    for p, m in zip(m_quantiles, m_levels):
        y = post.β0 + post.β1 * xi + post.β2 * xi * m + post.β3 * m
        region = y.quantile([0.025, 0.5, 0.975], dim="sample")
        ax.fill_between(
            xi,
            region.sel(quantile=0.025),
            region.sel(quantile=0.975),
            alpha=0.2,
            color=scalarMap.to_rgba(m),
            edgecolor="w",
        )
        ax.plot(
            xi,
            region.sel(quantile=0.5),
            color=scalarMap.to_rgba(m),
            linewidth=2,
            label=f"{p*100}th percentile of area",
        )

    ax.legend(fontsize=9)
    ax.set(xlabel="residents per unit area", ylabel="unique residents per unit area")
    return ax


def plot_moderation_effect(result, m, m_quantiles, scalarMap, ax=None):
    """Spotlight graph"""

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    post = az.extract(result)

    # calculate 95% CI region and median
    xi = xr.DataArray(np.linspace(np.min(m), np.max(m), 20), dims=["x_plot"])
    rate = post.β1 + post.β2 * xi
    region = rate.quantile([0.025, 0.5, 0.975], dim="sample")

    ax.fill_between(
        xi,
        region.sel(quantile=0.025),
        region.sel(quantile=0.975),
        alpha=0.2,
        color="k",
        edgecolor="w",
    )

    ax.plot(xi, region.sel(quantile=0.5), color="k", linewidth=2)

    # plot points at each percentile of m
    percentile_list = np.array(m_quantiles) * 100
    m_levels = np.percentile(m, percentile_list)
    for p, m in zip(percentile_list, m_levels):
        ax.plot(
            m,
            np.mean(post.β1) + np.mean(post.β2) * m,
            "o",
            c=scalarMap.to_rgba(m),
            markersize=10,
            label=f"{p}th percentile of area",
        )

    ax.legend(fontsize=9)

    ax.set(
        title="Spotlight graph",
        xlabel="$area$",
        ylabel=r"$\beta_1 + \beta_2 \cdot area$",
    )

def make_all_plots(trace, scalarMap):
    m_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    az.plot_trace(trace, figsize=(16, 22))

    az.plot_pair(
        trace,
        marginals=True,
        point_estimate="median",
        figsize=(12, 12),
        scatter_kwargs={"alpha": 0.01},
    )

    az.plot_posterior(trace, var_names=["β1", "β2", "β3"], figsize=(14, 4))

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data(trace.constant_data.x.values, trace.constant_data.m.values, trace.observed_data.y.values, scalarMap, ax=ax)

    posterior_prediction_plot(trace, trace.constant_data.x.values, trace.constant_data.m.values, m_quantiles, scalarMap, ax=ax)
    ax.set_title("Data and posterior prediction")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_moderation_effect(trace, trace.constant_data.m.values, m_quantiles, scalarMap, ax[0])
    az.plot_posterior(trace, var_names="β2", ax=ax[1])
