import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np 
import arviz as az
from typing import Tuple, Dict
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import xarray as xr
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from sqlalchemy import select, MetaData, Table, create_engine, select, func, and_, inspect

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Executable, ClauseElement

import operator

from hs_models.constants import (
    FOOTFALL_COUNTS_TABLE,
    HEX_GEOM_TABLE,
    HS_TABLE,
    TC_TABLE,
    BID_TABLE,
    HEX_AREA
)

import os

from dotenv import load_dotenv

load_dotenv()

sns.set_theme(style="ticks")

project_root=os.getenv("PROJ_ROOT")
if project_root is None:
    raise ValueError("PROJ_ROOT not found in .env file")


def plot_df(df, filename, title_string='', height=1200, width=400):
    formatted_values = []
    for col in df.columns:
        # Check if the column is a float type
        if pd.api.types.is_float_dtype(df[col]):
            formatted_values.append(df[col].round(2))
        
        else:
            formatted_values.append(df[col])
            
    fig = go.Figure(data=[go.Table(
        # Custom Header Styling
        header=dict(
            values=list(df.columns),
            fill_color='#1f77b4',     
            font=dict(color='white', size=14, family="Helvetica"),
            align='center',
            height=35
        ),
        # Custom Cells Styling
        cells=dict(
            values=formatted_values,
            fill_color=[['#f8f9fa', '#ffffff'] * 3], 
            font=dict(color='black', size=12, family="Helvetica"),
            align='center',
            height=30
        )
    )])

    fig.update_layout(    
        width=width, height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text=f"<b>{title_string}</b>",
            font=dict(size=18, family="Helvetica", color="#333333"),
            x=0.5,
            xanchor='center',
            y=0.96     
        )
    )

    
    fig.write_image(filename, width=width, height=height)

    fig.show()

    return fig

def zscore(s):
    if s.std() == 0:
        return 0
    return (s - s.mean()) / s.std()

def evaluate_dedupe_mape_threshold(y_test, y_pred, threshold=100):
    m = y_test >= threshold
    return mean_absolute_percentage_error(y_test[m], y_pred[m])

def load_data_long(dup_threshold=100000, interaction_cols = []):
    file_name = os.getenv("COUNT_DATA_FILE")
    area_file = os.getenv("AREA_FILE")

    df_long = pd.read_csv(file_name, parse_dates=['count_date'])
    area_df = pd.read_csv(area_file) 
    area_df['area'] = area_df['area'] / 1e6

    df_long['poi_nuid']=df_long['poi_type'] + '_' + df_long['poi_id'].astype(str)
    area_df['poi_nuid'] = area_df['poi_type'] + '_' + area_df['poi_id'].astype(str)
    area_df = area_df.groupby('poi_nuid')['area'].first().reset_index()

    df_long.rename(columns={
        'resident': 'duplicated_residents',
        'worker': 'duplicated_workers',
        'visitor': 'duplicated_visitors',
        'total_unique_domestic_visitors': 'total_unique_visitors'
    }, inplace=True)

    # add area data to main df
    df_long = df_long.merge(area_df[['poi_nuid', 'area']], on='poi_nuid', how='inner')

    df_long["id"] = df_long.index

    df_long = pd.wide_to_long(
        df_long,
        stubnames=["total_unique", "duplicated"],
        i="id",
        j="count_type",
        sep='_',
        suffix=r"\w+"
    )

    df_long = df_long.reset_index().drop(columns=['id'])
    df_long.dropna(inplace=True)

    df_long['area_hex'] = df_long['area'] / HEX_AREA

    df_long['count_type'] = df_long['count_type'].astype('category')
    df_long['caz_inner_outer'] = df_long['caz_inner_outer'].astype('category')
    df_long['poi_type'] = df_long['poi_type'].astype('category')
    df_long['time_indicator'] = df_long['time_indicator'].astype('category')
    df_long['poi_nuid'] = df_long['poi_nuid'].astype('category')
    
    df_long['year'] = df_long['count_date'].dt.year
    df_long['month'] = df_long['count_date'].dt.month
    df_long['day'] = df_long['count_date'].dt.day
    df_long['day_of_week'] = df_long['count_date'].dt.dayofweek
    df_long['hour'] = df_long['count_date'].dt.hour  
    df_long['is_weekend'] = (df_long['count_date'].dt.dayofweek >= 5).astype(int)

    df_long['timestamp'] = df_long['count_date'].astype('int64') // 10**9

    df_long['count_date'] = pd.to_datetime(df_long['count_date'])

    df_long['dup_zscored_by_area'] = df_long.groupby('poi_nuid')['duplicated'].transform(zscore)

    df_long = df_long[(df_long['duplicated'] < dup_threshold) & (df_long['duplicated'] > 0)]

    # add interaction terms
    if len(interaction_cols) > 0:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interactions = poly.fit_transform(df_long[interaction_cols])
        interaction_names = poly.get_feature_names_out(interaction_cols)
        df_interactions = pd.DataFrame(interactions, columns=interaction_names, index=df_long.index)
        df_long = pd.concat([df_long, df_interactions.drop(columns=interaction_cols)], axis=1)

    return df_long


def join_pois_from_db(df):

    unique_highstreets = df.loc[
            df['poi_nuid'].str.startswith('highstreets'), 'poi_nuid',
        ].str.slice(12).unique().astype(int).tolist()
    unique_tcs = df.loc[
            df['poi_nuid'].str.startswith('towncentres'), 'poi_nuid',
        ].str.slice(12).unique().astype(int).tolist()
    unique_bids = df.loc[
            df['poi_nuid'].str.startswith('bids'), 'poi_nuid',
        ].str.slice(5).unique().astype(int).tolist()

    database = os.getenv("PG_DATABASE")
    username = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")

    engine = create_engine(
        f"postgresql+psycopg2://{username}:{password}@"
        f"{host}:{port}/{database}"
    )

    metadata = MetaData()
    hs_table = Table(HS_TABLE, metadata, autoload_with=engine)
    tc_table = Table(TC_TABLE, metadata, autoload_with=engine)
    bid_table = Table(BID_TABLE, metadata, autoload_with=engine)

    query_hs = select(hs_table).where(hs_table.c.highstreet_id.in_(unique_highstreets))
    query_tc = select(tc_table).where(tc_table.c.tc_id.in_(unique_tcs))
    query_bid = select(bid_table).where(bid_table.c.bid_id.in_(unique_bids))

    with engine.connect() as conn:
        df_hs   = gpd.read_postgis(query_hs, conn)
        df_tc   = gpd.read_postgis(query_tc, conn)
        df_bid  = gpd.read_postgis(query_bid, conn)

    return df_hs, df_tc, df_bid

def get_area_geometry_from_database():

    database = os.getenv("PG_DATABASE")
    username = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")

    engine = create_engine(
        f"postgresql+psycopg2://{username}:{password}@"
        f"{host}:{port}/{database}"
    )

    metadata = MetaData()
    hs_table = Table(HS_TABLE, metadata, autoload_with=engine)
    tc_table = Table(TC_TABLE, metadata, autoload_with=engine)
    bid_table = Table(BID_TABLE, metadata, autoload_with=engine)

    query_hs = select(hs_table)
    query_tc = select(tc_table)
    query_bid = select(bid_table)

    with engine.connect() as conn:
        df_hs   = gpd.read_postgis(query_hs, conn)
        df_tc   = gpd.read_postgis(query_tc, conn)
        df_bid  = gpd.read_postgis(query_bid, conn)

    return df_hs, df_tc, df_bid


def get_hex_geometry():

    database = os.getenv("PG_DATABASE")
    username = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")

    engine = create_engine(
        f"postgresql+psycopg2://{username}:{password}@"
        f"{host}:{port}/{database}"
    )

    metadata = MetaData()
    hex_table = Table(HEX_GEOM_TABLE, metadata, autoload_with=engine)
    query_hex = select(hex_table)
    with engine.connect() as conn:
        df_footfall  = gpd.read_postgis(query_hex, conn)

    return df_footfall


OPERATOR_MAP = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "like": lambda col, val: col.like(val),
    "ilike": lambda col, val: col.ilike(val),
}

def build_sql_query(table, params: dict):
    """
    Dynamically constructs a SQLAlchemy 2.0 executable select statement 
    from a structured query parameter dictionary for a Core Table.

    Parameters
    ----------
    table : sqlalchemy.sql.schema.Table
        The target SQLAlchemy Core Table object to query against.
    params : dict
        A structured dictionary containing the query configurations. 
        Supported top-level keys include:

        * 'select' : list of str, optional
            A list of column names to retrieve. If both 'select' and 
            'aggregations' are omitted, defaults to selecting all columns (`SELECT *`).
            Example: ["id", "name", "email"]

        * 'where' : list of tuples, optional
            A list of filtering conditions evaluated as an implicit logical `AND`. 
            Each filter must be a tuple of exactly three elements: 
            `(column_name: str, operator: str, value: Any)`.
            Supported operators: "==", "!=", ">", ">=", "<", "<=", "like", "ilike".
            Example: [("age", ">=", 18), ("status", "==", "active")]

        * 'aggregations' : list of tuples, optional
            A list of SQL functions to apply to columns. Each item must be a 
            tuple of exactly three elements: `(column_name: str, function_name: str, alias: str)`.
            Common functions include: "count", "sum", "avg", "min", "max".
            Example: [("id", "count", "total_users"), ("salary", "avg", "average_pay")]

        * 'group_by' : list of str, optional
            A list of column names used to group rows for aggregate queries. 
            Example: ["department", "status"]

        * 'limit' : int, optional
            The maximum number of records to return. Must be a positive integer.
            Example: 50

        * 'offset' : int, optional
            The number of rows to skip before starting to return records. 
            Example: 100

    Returns
    -------
    sqlalchemy.sql.expression.Select
        An executable SQLAlchemy 2.0 select statement object ready to be 
        passed to `session.execute()`.

    Raises
    ------
    KeyError
        If a column name specified in 'select', 'where', 'aggregations', 
        or 'group_by' does not exist in the provided table configuration.

    Examples
    --------
        >>> payload = {
        ...     "select": ["department"],
        ...     "where": [("hire_date", ">=", "2023-01-01"), ("status", "==", "Active")],
        ...     "aggregations": [("id", "count", "total")],
        ...     "group_by": ["department"],
        ...     "limit": 10
        ... }
        >>> stmt = build_dynamic_table_query(employees_table, payload)
    """
    select_items = []
    
    if "select" in params:
        for col_name in params["select"]:
            select_items.append(table.c[col_name])
            
    if "aggregations" in params:
        for col_name, func_type, alias in params["aggregations"]:
            db_col = table.c[col_name]
            # Fetch the sql function dynamically (e.g., func.count, func.sum)
            sql_func = getattr(func, func_type)(db_col)
            if alias:
                sql_func = sql_func.label(alias)
            select_items.append(sql_func)
            
    if not select_items:
        stmt = select(table)
    else:
        stmt = select(*select_items)

    if "where" in params:
        where_clauses = []
        for col_name, op_str, value in params["where"]:
            db_col = table.c[col_name]
            op_func = OPERATOR_MAP.get(op_str)
            
            if op_func:
                where_clauses.append(op_func(db_col, value))
        
        if where_clauses:
            stmt = stmt.where(and_(*where_clauses))

    if "group_by" in params:
        group_cols = [table.c[col] for col in params["group_by"]]
        stmt = stmt.group_by(*group_cols)

    if "limit" in params:
        stmt = stmt.limit(params["limit"])
    if "offset" in params:
        stmt = stmt.offset(params["offset"])

    return stmt


def get_footfall_data(query_dict: dict):

    database = os.getenv("PG_DATABASE")
    username = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")

    engine = create_engine(
        f"postgresql+psycopg2://{username}:{password}@"
        f"{host}:{port}/{database}"
    )

    metadata = MetaData()
    hex_table = Table(FOOTFALL_COUNTS_TABLE, metadata, autoload_with=engine)
    
    query_footfall = build_sql_query(hex_table, query_dict)
        
    with engine.connect() as conn:
        df_fotfall  = pd.read_sql(query_footfall, conn)

    return df_fotfall


def get_db_metadata():

    database = os.getenv("PG_DATABASE")
    username = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")

    engine = create_engine(
        f"postgresql+psycopg2://{username}:{password}@"
        f"{host}:{port}/{database}"
    )

    metadata = MetaData()

    # Reflect all tables found in the database schema
    metadata.reflect(bind=engine)

    return metadata, engine



def error_vs_area_plot(df, y_true, y_pred, area_col='area_hex'):

    df = df.copy()

    df['error'] = 100*np.abs((y_true - y_pred)) / y_true

    df_error = df[[
        'poi_nuid', 
        area_col, 
        'count_type', 
        'time_indicator', 
        'error', 
    ]]

    df_error = df_error[np.isfinite(df_error['error'])]

    poi_error = df_error.groupby(['poi_nuid', 'count_type', 'time_indicator'])[[area_col, 'error']].mean().reset_index()

    poi_error['area_bin'] = pd.qcut(poi_error[area_col], q=40)
    poi_error['area_bin'] = poi_error['area_bin'].apply(lambda x: x.mid)

    g = error_vs_area(poi_error)

    return g, poi_error


def error_vs_area(poi_error):
    g=sns.relplot(
        data=poi_error,
        x='area_bin',
        y='error',
        # hue='time_indicator',
        kind='line',     
        errorbar='ci',   
        estimator='median',
        facet_kws = {'sharey': False}
    )
    g.refline(y=15, color='red', linestyle='--', linewidth=1, label='15% error')
    g.refline(y=30, color='green', linestyle='--', linewidth=1, label='30% error')

    g.set(yscale='log')
    g.set(xscale='log')
    g.set(ylim=(3, 1000))
    g.add_legend()
    g.axes[0][0].set_xlabel('Area [hexes]')
    g.axes[0][0].set_ylabel('Avg. absolute % error')

    return g

def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def load_footfall_dedupe_data(
    file_name, area_file
):

    # get object and file (key) from bucket
    observation_df = pd.read_csv(
        file_name, 
        parse_dates=['count_date'],
        low_memory=False) 

    observation_df.rename(
        columns={'total_unique_domestic_visitors': 'total_unique_visitors'},
        inplace=True,
    )

    # get object and file (key) from bucket
    area_df = pd.read_csv(area_file) 

    # Keep only the all-day numbers
    # day_mask = observation_df['time_indicator'] == 'DAY'
    # observation_df_filt = observation_df[day_mask]

    day_df = (
        observation_df[observation_df['time_indicator'] == 'DAY']
        .dropna(subset=['poi_uid', 'count_date'])
        .drop(columns='time_indicator')
        .rename(columns={
            'visitor': 'visitor_day', 
            'worker': 'worker_day', 
            'resident': 'resident_day', 'total_unique_workers': 'total_unique_workers_day', 'total_unique_residents': 'total_unique_residents_day',
            'total_unique_visitors': 'total_unique_visitors_day',
            'avg_dwell_time': 'avg_dwell_time_day'})
    )

    am_df = (
        observation_df[observation_df['time_indicator'] == 'AM']
        .dropna(subset=['poi_uid', 'count_date'])
        .drop(columns='time_indicator')
        .rename(columns={
            'visitor': 'visitor_am', 
            'worker': 'worker_am', 
            'resident': 'resident_am', 
            'total_unique_workers': 'total_unique_workers_am', 'total_unique_residents': 'total_unique_residents_am' ,
            'total_unique_visitors': 'total_unique_visitors_am','avg_dwell_time': 'avg_dwell_time_am'
            })
    )

    pm_df = (
        observation_df[observation_df['time_indicator'] == 'PM']
        .dropna(subset=['poi_uid', 'count_date'])
        .drop(columns='time_indicator')
        .rename(columns={
            'visitor': 'visitor_pm', 
            'worker': 'worker_pm', 
            'resident': 'resident_pm', 
            'total_unique_workers': 'total_unique_workers_pm', 'total_unique_residents': 'total_unique_residents_pm',
            'total_unique_visitors': 'total_unique_visitors_pm','avg_dwell_time': 'avg_dwell_time_pm'
            })
    )

    observation_df=day_df.merge(am_df, on=[
            'poi_uid', 
            'count_date', 
            'poi_name', 
            'poi_id', 
            'poi_type', 
            'caz_inner_outer',
        ], 
        how='inner', 
        suffixes=('_day', '_am'),
    )

    observation_df=observation_df.merge(pm_df, on=['poi_uid', 'count_date', 'poi_name', 'poi_id', 'poi_type', 'caz_inner_outer'], how='inner', suffixes=('_day', '_pm'))

    observation_df['poi_nuid']=observation_df['poi_type'] + '_' + observation_df['poi_id'].astype(str)

    observation_df = observation_df[['poi_nuid'] + list(observation_df.columns[:-1])]

    area_df['poi_nuid'] = area_df['poi_type'] + '_' + area_df['poi_id'].astype(str)

    area_df = area_df[['poi_nuid'] + list(area_df.columns[:-1])]


    # take the mean area value for each area (in case there's noise?)
    # areas are measured in meters squared, rescale to use kilometers squared as this give more manageable numbers
    area_df = area_df.groupby('poi_nuid')['area'].mean().reset_index() 

    area_df['area'] = area_df['area'] / 1e6

    # add area data to main df
    observation_df = observation_df.merge(area_df[['poi_nuid', 'area']], on='poi_nuid', how='left')

    observation_df = observation_df[~observation_df['area'].isna()]

    # for now we remove the very largest areas which show less predictable behaviour
    # observation_df = observation_df[observation_df['area'] < 0.4]

    # create area bins so we can take a stratified sample across the range of area sizes
    observation_df['area_bin'] = pd.qcut(observation_df['area'], q=6)

    stats_dfs = []

    for count_type in ['worker', 'resident', 'visitor']:

        for time_indicator in ['day', 'am', 'pm']:
                
            # for now we remove any areas that have zero unique residents, workers, or visitors
            observation_df = observation_df[observation_df[f'total_unique_{count_type}s_{time_indicator}'] > 0]

            # we will model counts per unit area as these have a more manageable scale
            observation_df[f'{count_type}s_per_area_{time_indicator}'] = (
                observation_df[f'{count_type}_{time_indicator}'] / observation_df['area']
            )

            observation_df[f'unique_{count_type}s_per_area_{time_indicator}'] = (
                observation_df[f'total_unique_{count_type}s_{time_indicator}'] / observation_df['area']
            ) 

            # compute best fit lines between observed and de-duped  counts for all areas
            res = (
                observation_df
                .groupby('poi_nuid')
                .apply(
                    lambda x: fit_line(
                        x[f'{count_type}s_per_area_{time_indicator}'], 
                        x[f'unique_{count_type}s_per_area_{time_indicator}']), 
                        include_groups=False,
                    )
            )

            stats_df = pd.DataFrame(res.tolist(), index=res.index, columns=[f'slope_{count_type}', f'intercept_{count_type}']).reset_index()

            stats_df.rename(columns={
                f'slope_{count_type}': 'slope', 
                f'intercept_{count_type}': 'intercept',
            }, inplace=True)

            stats_df['count_type'] = count_type
            stats_df['count_time'] = time_indicator

            stats_dfs.append(stats_df)
            

    stats_df = pd.concat(stats_dfs)
    stats_df = stats_df.merge(area_df[['poi_nuid', 'area']], on='poi_nuid', how='left')
    stats_df['area_bin'] = pd.qcut(
        stats_df['area'], 
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        )


    return observation_df, stats_df


def get_sample_of_footfall_dedupe_data(
    observation_df_filt,
    n_sample_areas = 12,
    n_obs_per_area = 100
):
    sample_areas = (
        observation_df_filt
        .groupby('area_bin', observed=False)['poi_uid']
        .unique()
        .apply(lambda x: np.random.choice(
            x, 
            size=n_sample_areas, 
            replace=False,
            )).explode()
    )

    sample_data = (
        observation_df_filt[observation_df_filt['poi_uid']
        .isin(sample_areas)]
        .sort_values('count_date', ascending=False)
        .groupby('poi_uid')
        .apply(lambda x: x.sample(n_obs_per_area, replace=True), include_groups=False)
        .drop(columns=['poi_id', 'area_bin'])
        .reset_index()
    )

    sample_data.dropna(inplace=True)

    sample_data['area_ids'] = sample_data.groupby('poi_uid').ngroup()

    return sample_data

# fit a line to each di vs oi relationship
def fit_line(x, y):
    # Calculate the means of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the slope (m)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    if denominator == 0:
        m = np.nan
    else:
        m = numerator / denominator

    # Calculate the y-intercept (b)
    b = y_mean - m * x_mean

    return m, b

def plot_data_examples(data, nx=6, ny=6):

    fig, axes = plt.subplots(
        nx, 
        ny, 
        figsize=(32, 19), 
        sharey=False, 
        sharex=False, 
        dpi=300, 
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.075, 
        right=0.975, 
        bottom=0.075, 
        top=0.925, 
        wspace=0.2,
    )

    axes_flat = axes.ravel()
    m, b, a = [], [], []

    for i, area_id in enumerate(data["poi_uid"].unique()):
        
        idx = data.index[data["poi_uid"] == area_id].tolist()
        resident = data.loc[idx, "residents_per_area"].values
        unique_residents = data.loc[idx, "unique_residents_per_area"].values

        if i < nx*ny:
            ax = axes_flat[i]
            # Plot observed data points
            ax.scatter(resident, unique_residents, color="C0", ec="black", alpha=0.7)

            # Add a title
            ax.set_title(f"area_id: {area_id}", fontsize=12)
    
    fig.text(0.5, 0.02, "residents", fontsize=14)
    fig.text(0.01, 0.5, "unique residents", rotation=90, fontsize=14, va="center")

    return axes

def center_data(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Center data by subtracting mean and dividing by standard deviation.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data to be centered
        
    Returns:
    --------
    Tuple[np.ndarray, Dict]
        centered_data: Centered and scaled data (mean=0, std=1)
        scaler_info: Dictionary with scaling parameters for uncentering
    """
    mean = np.mean(data)
    std = np.std(data)
    
    # Avoid division by zero
    if std == 0:
        std = 1.0
    
    centered_data = (data - mean) / std
    
    scaler_info = {
        'mean': mean,
        'std': std,
        'original_shape': data.shape,
        'original_dtype': data.dtype
    }
    
    return centered_data, scaler_info

def uncenter_data(
        centered_data: np.ndarray, 
        scaler_info: Dict,
    ) -> np.ndarray:
    """
    Reverse the centering transformation to return to original scale.
    
    Parameters:
    -----------
    centered_data : np.ndarray
        Centered data (mean=0, std=1)
    scaler_info : Dict
        Dictionary containing scaling parameters from center_data()
        
    Returns:
    --------
    np.ndarray
        Data in original scale
    """
    return centered_data * scaler_info['std'] + scaler_info['mean']


# some helper plotting functions

def make_scalarMap(m):
    """Create a Matplotlib `ScalarMappable` so we can use a consistent colormap across both data points and posterior predictive lines. We can use `scalarMap.cmap` to use as a colormap, and `scalarMap.to_rgba(moderator_value)` to grab a colour for a given moderator value."""
    return ScalarMappable(norm=Normalize(vmin=np.min(m), vmax=np.max(m)), cmap="viridis")


def plot_data(x, moderator, y, scalarMap, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()

    h = ax.scatter(x, y, c=moderator, cmap=scalarMap.cmap)
    ax.set(xlabel="x", ylabel="y")

    # colourbar for moderator
    cbar = fig.colorbar(h)
    cbar.ax.set_ylabel("area")
    return ax


def posterior_prediction_plot(result, x, moderator, m_quantiles, scalarMap, ax=None):
    """Plot posterior predicted `y`"""
    if ax is None:
        _, ax = plt.subplots(1, 1)

    post = az.extract(result)
    xi = xr.DataArray(np.linspace(np.min(x), np.max(x), 20), dims=["x_plot"])
    m_levels = result.constant_data["m"].quantile(m_quantiles).rename({"quantile": "m_level"})

    for p, m in zip(m_quantiles, m_levels):
        y = post.β0 + post.β1 * xi + post.β2 * xi * m + post.β3 * m
        region = y.quantile([0.025, 0.5, 0.975], dim="sample")
        ax.fill_between(
            xi,
            region.sel(quantile=0.025),
            region.sel(quantile=0.975),
            alpha=0.2,
            color=scalarMap.to_rgba(m),
            edgecolor="w",
        )
        ax.plot(
            xi,
            region.sel(quantile=0.5),
            color=scalarMap.to_rgba(m),
            linewidth=2,
            label=f"{p*100}th percentile of area",
        )

    ax.legend(fontsize=9)
    ax.set(xlabel="residents per unit area", ylabel="unique residents per unit area")
    return ax


def plot_moderation_effect(result, m, m_quantiles, scalarMap, ax=None):
    """Spotlight graph"""

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    post = az.extract(result)

    # calculate 95% CI region and median
    xi = xr.DataArray(np.linspace(np.min(m), np.max(m), 20), dims=["x_plot"])
    rate = post.β1 + post.β2 * xi
    region = rate.quantile([0.025, 0.5, 0.975], dim="sample")

    ax.fill_between(
        xi,
        region.sel(quantile=0.025),
        region.sel(quantile=0.975),
        alpha=0.2,
        color="k",
        edgecolor="w",
    )

    ax.plot(xi, region.sel(quantile=0.5), color="k", linewidth=2)

    # plot points at each percentile of m
    percentile_list = np.array(m_quantiles) * 100
    m_levels = np.percentile(m, percentile_list)
    for p, m in zip(percentile_list, m_levels):
        ax.plot(
            m,
            np.mean(post.β1) + np.mean(post.β2) * m,
            "o",
            c=scalarMap.to_rgba(m),
            markersize=10,
            label=f"{p}th percentile of area",
        )

    ax.legend(fontsize=9)

    ax.set(
        title="Spotlight graph",
        xlabel="$area$",
        ylabel=r"$\beta_1 + \beta_2 \cdot area$",
    )

def make_all_plots(trace, scalarMap):
    m_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    az.plot_trace(trace, figsize=(16, 22))

    az.plot_pair(
        trace,
        marginals=True,
        point_estimate="median",
        figsize=(12, 12),
        scatter_kwargs={"alpha": 0.01},
    )

    az.plot_posterior(trace, var_names=["β1", "β2", "β3"], figsize=(14, 4))

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data(trace.constant_data.x.values, trace.constant_data.m.values, trace.observed_data.y.values, scalarMap, ax=ax)

    posterior_prediction_plot(trace, trace.constant_data.x.values, trace.constant_data.m.values, m_quantiles, scalarMap, ax=ax)
    ax.set_title("Data and posterior prediction")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_moderation_effect(trace, trace.constant_data.m.values, m_quantiles, scalarMap, ax[0])
    az.plot_posterior(trace, var_names="β2", ax=ax[1])


def plot_sample_data(X, y, filename=None):
    X_all = X.copy()
    X_all['deduped counts'] = y.copy()
    X_all.rename(columns={'total_counts': 'total counts'}, inplace=True)

    n_x=3 
    n_y = 3
    n=n_x * n_y

    b1 = -0.005
    b2 = 0.185

    area_bins = np.linspace(X_all['area_size'].min(), X_all['area_size'].max(), n+1)

    X_all['area_bin'] = pd.cut(X_all['area_size'], area_bins)

    sample_areas = X_all.groupby('area_bin', observed=False).apply(lambda x: x.sample(1), include_groups=False)['area_id']

    X_plot = X_all[X_all['area_id'].isin(sample_areas)].sort_values('area_bin')

    def scatter_w_fit(x, y, z, **kwargs):
        sns.scatterplot(x=x, y=y, **kwargs)
        xmin, xmax = (
            x.min(), 
            x.max()
        )
        ymin, ymax = (
            b1*x.min() + b2*x.min()*z.mean(), 
            b1*x.max() + b2*x.max()*z.mean(),
        )
        plt.plot((xmin, xmax), (ymin, ymax), '-k')

    g = sns.FacetGrid(X_plot, col='area_id', hue='area_bin', palette='flare', col_wrap=3)
    g.map(scatter_w_fit, 'total counts', 'deduped counts', 'area_size')

    if filename:
        try:
            g.savefig(filename)
        except:
            print(f'could not save figure to file {filename}')

    return g


def plot_data_prior_posterior(model):

    df = pd.DataFrame(
        {
            'total_counts': model.idata.fit_data.total_counts,
            'area': model.idata.fit_data.area_size,
            'y obs': model.idata.fit_data.y,
            'y prior': model.idata.prior_predictive.y.mean(dim=('chain', 'draw')),
            'y posterior': model.idata.posterior_predictive.y.mean(dim=('chain', 'draw')),
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    plot_quantities = ['obs', 'prior', 'posterior']
    for idx, pq in enumerate(plot_quantities):
        sns.scatterplot(df, x='total_counts', y=f'y {pq}', hue='area', ax=axes[idx])
        axes[idx].set_title(f'{pq}')

    return fig, axes


def get_table_indexes(engine, table_name, do_print=False):

    inspector = inspect(engine)
    indexes = inspector.get_indexes(table_name)

    if do_print:
        print(f"--- Indexes on table '{FOOTFALL_COUNTS_TABLE}' ---")
        if not indexes:
            print("No indexes found!")

        else:
            for index in indexes:
                print(f"Index Name: {index['name']}")
                print(f"  Columns:  {index['column_names']}")
                print(f"  Unique?:  {index['unique']}")
                print("-" * 30)

    return indexes

class ExplainAnalyze(Executable, ClauseElement):
    def __init__(self, stmt):
        self.stmt = stmt

@compiles(ExplainAnalyze, "postgresql")
def compile_explain_analyze(element, compiler, **kw):
    # This prepends the exact syntax dynamically at runtime
    return f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {compiler.process(element.stmt, **kw)}"
