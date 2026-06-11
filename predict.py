# %% Imports and setup
import pandas as pd 
import hs_models.utils as util 
import  joblib

import lightgbm as lgb
from supertree import SuperTree

from dotenv import load_dotenv
load_dotenv()

# %% Load data

df_long = util.load_data_long()

#%% [markdown]
# The data has already been aggregated by summing across hexes and time 
#  
# Each row represents the aggregate count across hexes covering an area of interest 
#  
# during one of three time periods (AM, PM, DAY) on a given date  

# the model has been trained on such data and the count_time (AM, PM, DAY) is an input feature 
#  
# We could alternatively train a model where the number of bins, or the specific bins themselves
#   
# are the inputs, which would allow us to dedupe over arbitrary areas and time periods  

# %% Look at the data

df_long.head()

#%% [markdown]
# Load the trained models - these are stored as pkl files
# 
# To run these models would require an identical python environment and the pkl files
# 
# There are various alternative ways to deploy - for example ONNX is a possibility 

# %% Load the trained model and preprocessing

poly_load = joblib.load('./models/poly_interactions.pkl')

bst_final = lgb.Booster(model_file='./models/ldt.txt')

# %% Add some simple features

def zscore(s):
    # Handle groups with no variance to avoid NaNs
    if s.std() == 0:
        return 0
    return (s - s.mean()) / s.std()

df_long['dup_zscored_by_area'] = df_long.groupby('poi_nuid')['duplicated'].transform(zscore)

interaction_cols = ['area_hex', 'duplicated']
interactions = poly_load.transform(df_long[interaction_cols])

interaction_names = poly_load.get_feature_names_out(interaction_cols)
df_interactions = pd.DataFrame(interactions, columns=interaction_names, index=df_long.index)

df_long = pd.concat([df_long, df_interactions.drop(columns=interaction_cols)], axis=1)

X = df_long.drop(columns=['poi_name', 'poi_id', 'poi_uid', 'total_unique', 'count_date'])

y = df_long['total_unique']


# %% Visualise the tre

st = SuperTree(
    bst_final, 
    X, 
    y, 
    X.columns.to_list(),
    "deduplicated count"
)

# Visualize the tree
st.show_tree(which_tree=0)


# %% predict

predictions = bst_final.predict(X)


# %%

score = util.evaluate_dedupe_mape_threshold(y, predictions)

g = util.error_vs_area_plot(X, y, predictions)


# %%


