import os
from pathlib import Path
import typer

from hs_models.utils import (
    load_footfall_dedupe_data, 
    get_sample_of_footfall_dedupe_data,
    plot_sample_data
)

from dotenv import load_dotenv
load_dotenv()

env_var_names = [
    'SAMPLE_X', 
    'SAMPLE_Y', 
    'SAMPLE_DIR', 
    'SAMPLE_FIG',
    'MODEL_FIT_FIGS', 
    'DATA_BUCKET',
    'COUNT_DATA_FILE',
    'AREA_FILE',
]

env_vars = {}
for dir_check in env_var_names:
    env_vars[dir_check]=os.getenv(dir_check)
    if env_vars[dir_check] is None:
        raise ValueError(f"{dir_check} not found in .env file")

def main(
        count_type: str = 'worker',
        count_time: str = 'day'
    ):

    Path(env_vars['SAMPLE_DIR']).mkdir(parents=True, exist_ok=True)
    Path(env_vars['MODEL_FIT_FIGS']).mkdir(parents=True, exist_ok=True)

    observation_df_filt, stats_df = load_footfall_dedupe_data(
        env_vars['DATA_BUCKET'],
        env_vars['COUNT_DATA_FILE'],
        env_vars['AREA_FILE']
    )

    sample_data = get_sample_of_footfall_dedupe_data(
        observation_df_filt,
        n_sample_areas = 36,
        n_obs_per_area = 50
    )

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

    X.to_csv(os.path.join(env_vars['SAMPLE_DIR'], env_vars['SAMPLE_X']), index=False)
    y.to_csv(os.path.join(env_vars['SAMPLE_DIR'], env_vars['SAMPLE_Y']), index=False)

    plot_sample_data(X, y, filename=os.path.join(env_vars['MODEL_FIT_FIGS'], env_vars['SAMPLE_FIG']))

if __name__ == "__main__":
    typer.run(main)