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

from hs_models.utils import (
    load_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
    load_9_models
)

from dotenv import load_dotenv

from hs_models.models import AreaCountInteraction1DPartPool

scale_df=pd.read_csv('./output/scale_factor_lookup.csv')

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
stats_df[(stats_df['count_type']=='resident') & (stats_df['area_bin'])]