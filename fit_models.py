import typer
from typing_extensions import Annotated, Literal
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import os
from pathlib import Path
from dotenv import load_dotenv

from hs_models.models import (
    LinearPoolB1,
    LinearPartPoolB1, 
    LinearPoolB1PoolB2,
    AreaCountInteraction1DPartPoolPositive,
)

from hs_models.utils import (
    plot_sample_data,
    plot_data_prior_posterior
)

def main(model: Annotated[Literal[
    'baseline',
    'partpool_b1',
    'pool_b1b2',
    'partpool_b1b2',
    'all'
], typer.Option()] = 'baseline'):

    match model:
        case 'baseline':
            models = [LinearPoolB1()]
        case 'partpool_b1':
            models = [LinearPartPoolB1()]
        case 'pool_b1b2':
            models = [LinearPoolB1PoolB2()]
        case 'partpool_b1b2':
            models = [AreaCountInteraction1DPartPoolPositive()]
        case 'all':
            models = [
                LinearPoolB1(),
                LinearPartPoolB1(),
                LinearPoolB1PoolB2(),
            ]

    load_dotenv()

    env_var_names = [
        'SAMPLE_X', 
        'SAMPLE_Y', 
        'SAMPLE_DIR', 
        'MODEL_DIR',
        'MODEL_FIT_FIGS'
    ]

    env_vars = {}
    for env_var in env_var_names:
        env_vars[env_var]=os.getenv(env_var)
        if env_vars[env_var] is None:
            raise ValueError(f"{env_var} not found in .env file")

    Path(env_vars['MODEL_DIR']).mkdir(parents=True, exist_ok=True)
    Path(env_vars['MODEL_FIT_FIGS']).mkdir(parents=True, exist_ok=True)
    
    # 0. load data
    X = pd.read_csv(os.path.join(env_vars['SAMPLE_DIR'], env_vars['SAMPLE_X']))
    y = pd.read_csv(os.path.join(env_vars['SAMPLE_DIR'], env_vars['SAMPLE_Y'])).squeeze()

    for idx, model in enumerate(models):
        print(f'Sampling from model {idx+1} of {len(models)}')

        # 1. Prior predictive checks
        prior_predictive_samples = model.sample_prior_predictive(X)

        # 2. Fit model
        model.fit(X, y)

        # 3. Posterior checks
        az.plot_trace(model.idata, figsize=(20,10))
        plt.savefig(os.path.join(
            env_vars['MODEL_FIT_FIGS'],
            f'{model._model_type}_trace.png'
        ))
        plt.close()

        az.summary(model.idata)
        plt.savefig(os.path.join(
            env_vars['MODEL_FIT_FIGS'],
            f'{model._model_type}_summary.png'
        ))
        plt.close()

        fig, _ = plot_data_prior_posterior(model)
        fig.savefig(os.path.join(
            env_vars['MODEL_FIT_FIGS'],
            f'{model._model_type}_data_prior_posterior.png'
        ))

        # 4. Save model
        model.save(os.path.join(env_vars['MODEL_DIR'], model._model_type))


if __name__ == "__main__":
    typer.run(main)