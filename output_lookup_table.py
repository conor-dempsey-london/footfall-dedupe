import arviz as az
import pymc as pm
import numpy as np
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
from scipy import stats 

from hs_models.utils import (
    load_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
)
from hs_models.models import AreaCountInteraction1DPartPool

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

count_types = ['worker', 'resident', 'visitor']
count_times = ['day', 'am', 'pm']

X = {}
y = {}
models = {}

for count_type in count_types:
    X_count_type = {}
    y_count_type = {}
    models_count_type = {}

    for count_time in count_times:
        X_count_type[count_time] = sample_data[
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

        y_count_type[count_time] = sample_data[f'total_unique_{count_type}s_day']

        models_count_type[count_time] = AreaCountInteraction1DPartPool()

    X[count_type] = X_count_type
    y[count_type] = y_count_type
    models[count_type] = models_count_type

areas = np.linspace(0, 0.5, 100)
scale_factor_dfs = []

Path("./models").mkdir(exist_ok=True)

for count_type in count_types:
    for count_time in count_times:
        print(f'Fitting model for: {count_type}s {count_time}')
        
        models[count_type][count_time].fit(
            X[count_type][count_time], 
            y[count_type][count_time],
        )

        fname = f'./models/parpool_{count_type}_{count_time}.nc'
        models[count_type][count_time].save(fname)

        map_b1 = models[count_type][count_time].idata.posterior.β1.mean(dim=['chain', 'draw'])
        map_mu_b2 = models[count_type][count_time].idata.posterior.mu_b.mean(dim=['chain', 'draw'])
        map_sigma_b2 = models[count_type][count_time].idata.posterior.sigma_b.mean(dim=['chain', 'draw'])

        rv = stats.Normal(mu=map_mu_b2.item(), sigma=map_sigma_b2.item())

        lower_beta2 = rv.icdf(0.05)
        mean_beta_2 = rv.icdf(0.5)
        upper_beta2 = rv.icdf(0.95)

        scale_factors_low   = map_b1.item() + lower_beta2 * areas
        scale_factors_mean  = map_b1.item() + mean_beta_2 * areas
        scale_factors_high  = map_b1.item() + upper_beta2 * areas

        scale_factor_dfs.append(
            pd.DataFrame(
                {
                    'Area (km2)': areas,
                    'factor_low': scale_factors_low,
                    'MAP scale factor': scale_factors_mean,
                    'factor_high': scale_factors_high,
                    'count_type': len(areas)*[count_type],
                    'count_time': len(areas)*[count_time],
                }
            )
        )


scale_factors = pd.concat(scale_factor_dfs)

Path("./output").mkdir(exist_ok=True)

print('\n', 'Saving scale factor lookup table to file...')
scale_factors.to_csv('./output/scale_factor_lookup.csv', index=False)

