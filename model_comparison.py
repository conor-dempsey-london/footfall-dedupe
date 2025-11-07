import typer
from typing import List
import pymc as pm
import arviz as az
import os
from dotenv import load_dotenv

from hs_models.models import (
    LinearPoolB1,
    LinearPartPoolB1, 
    LinearPoolB1PoolB2,
    LinearPartPoolB2,
)


def main():

    models = [
        LinearPoolB1(),
        LinearPartPoolB1(),
        LinearPoolB1PoolB2(),
        LinearPartPoolB2()
    ]

    load_dotenv()

    env_var_names = [
        'MODEL_DIR',
    ]

    env_vars = {}
    for env_var in env_var_names:
        env_vars[env_var]=os.getenv(env_var)
        if env_vars[env_var] is None:
            raise ValueError(f"{env_var} not found in .env file")

    for model in models:
        model = model.load(os.path.join(
            env_vars['MODEL_DIR'],
            model._model_type
        ))
        with model.model:
            if not 'log_likelihood' in model.idata.keys():
                pm.compute_log_likelihood(model.idata)

    df_comp_loo = az.compare({model._model_type: model.idata for model in models})

    df_comp_loo.to_csv('model_comparison.csv')
    

if __name__ == "__main__":
    typer.run(main)