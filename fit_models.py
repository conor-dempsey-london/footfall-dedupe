import typer
from typing_extensions import Annotated, Literal
import pandas as pd
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
    'partpool_b1b2'
], typer.Option()] = 'baseline'):

    match model:
        case 'baseline':
            model = LinearPoolB1()
        case 'partpool_b1':
            model = LinearPartPoolB1()
        case 'pool_b1b2':
            model = LinearPoolB1PoolB2()
        case 'partpool_b1b2':
            model = AreaCountInteraction1DPartPoolPositive()

    load_dotenv()

    env_var_names = [
        'SAMPLE_X', 
        'SAMPLE_Y', 
        'SAMPLE_DIR', 
        'MODEL_DIR',
    ]

    env_vars = {}
    for dir_check in env_var_names:
        env_vars[dir_check]=os.getenv(dir_check)
        if env_vars[dir_check] is None:
            raise ValueError(f"{dir_check} not found in .env file")

    Path(env_vars['MODEL_DIR']).mkdir(parents=True, exist_ok=True)
    
    # 0. load data
    X = pd.read_csv(os.path.join(env_vars['SAMPLE_DIR'], env_vars['SAMPLE_X']))
    y = pd.read_csv(os.path.join(env_vars['SAMPLE_DIR'], env_vars['SAMPLE_Y']))

    # 1. Prior predictive checks
    prior_predictive_samples = model.sample_prior_predictive(X)
    plot_sample_data(X, prior_predictive_samples.y.mean(dim=('sample')))

    # 2. Fit model
    model.fit(X, y)

    # 3. Posterior checks
    az.plot_trace(model.idata, figsize=(20,10))
    az.summary(model.idata)
    plot_data_prior_posterior(model)

    # 4. Save model
    model.save(os.path.join(env_vars['MODEL_DIR'], model._model_type))


if __name__ == "__main__":
    typer.run(main)