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

import matplotlib.pyplot as plt

from hs_models.utils import (
    load_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
    load_9_models
)

from hs_models.constants import HEX_AREA

from dotenv import load_dotenv

from hs_models.models import AreaCountInteraction1DPartPool

scale_df=pd.read_csv('../output/scale_factor_lookup.csv')

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

models = load_9_models()

# %%

sns.lineplot(
    scale_df, 
    x='Area (km2)', 
    y='MAP scale factor', 
    hue='count_type', 
    style='count_time',
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

ax=area_overcount_ratios.plot(x='Area (# hexes)', y='overcount_ratio_worker_day', style='o', logy=False, ylim=(0,55), xlim=(0,20))

ax.plot((0,50), (5,5))

plt.savefig(f'../figures/overcount/worker_day.png')


# %%

ax=area_overcount_ratios.plot(
    x='Area (# hexes)', 
    y='visitor_day_scale_factor', 
    style='o', logy=False, ylim=(0,1), xlim=(0,20))

ax.plot((0,50), (0.2, 0.2))

plt.savefig(f'../figures/overcount/visitor_day.png')


# %%

# compute the average overcount ratio in area_hexes bins from 0.5 to 5
area_overcount_ratios['area_bin'] = pd.cut(area_overcount_ratios['Area (# hexes)'], 40, labels=False, retbins=False)

area_overcount_ratios_binned = area_overcount_ratios.groupby('area_bin').mean()

area_overcount_ratios_binned['analytic_overcount_ratio'] = 5 + 6.4 / area_overcount_ratios_binned['Area (# hexes)'] ** 2

area_overcount_ratios_binned['simple_power'] = 12 / area_overcount_ratios_binned['Area (# hexes)'] ** 1.5

ax=area_overcount_ratios_binned.plot(x='Area (# hexes)', y= ['overcount_ratio_worker_day', 'analytic_overcount_ratio'], style='o')
ax.plot((0,50), (5,5))

ax.set_ylim((0, 20))
ax.set_xlim((0.0099, 50))

# %%





