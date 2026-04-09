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

env_var_names = [
    'MODEL_DIR',
]

env_vars = {}
for env_var in env_var_names:
    env_vars[env_var]=os.getenv(env_var)
    if env_vars[env_var] is None:
        raise ValueError(f"{env_var} not found in .env file")

sns.set_theme(style="ticks")

count_types = ['worker', 'resident', 'visitor']
count_times = ['day', 'am', 'pm']

model_base = AreaCountInteraction1DPartPool()

areas = np.logspace(-1.301029999566, 3.21, 2000)
scale_factor_dfs = []

Path("./models").mkdir(exist_ok=True)

for count_type in count_types:
    for count_time in count_times:
        print(f'Loading model for: {count_type}s {count_time}')
        
        fname = f'./models/parpool_{count_type}_{count_time}.nc'

        model = model_base.load(fname)

        map_b1 = model.idata.posterior.β1.mean(dim=['chain', 'draw'])
        map_mu_b2 = model.idata.posterior.mu_b.mean(dim=['chain', 'draw'])
        map_sigma_b2 = model.idata.posterior.sigma_b.mean(dim=['chain', 'draw'])

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

