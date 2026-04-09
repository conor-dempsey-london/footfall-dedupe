# %%
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
    LinearPartPoolB2PoolB1
)



models = [
    LinearPoolB1(),
    LinearPartPoolB1(),
    LinearPoolB1PoolB2(),
    LinearPartPoolB2(),
    LinearPartPoolB2PoolB1()
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

# %%

for model in models:
    model = model.load(os.path.join(
        env_vars['MODEL_DIR'],
        model._model_type
    ))
    pm.model_to_graphviz(model.model).render(f'./figures/model_graphs/{model._model_type}', format='png')

# %%
model = models[0].load(os.path.join(
        env_vars['MODEL_DIR'],
        models[0]._model_type
    ))

pm.model_to_graphviz(model.model)

# %%


