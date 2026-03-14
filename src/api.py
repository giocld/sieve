"""
FastAPI backend for Sieve NBA Analytics.
Exposes REST endpoints for player/team data and Plotly chart JSON.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Optional, List, Any
import pandas as pd
import numpy as np
import traceback
import json
import plotly.io as pio

from . import data_processing
from .data_processing import (
    get_season_list, fetch_standings, fetch_nba_advanced_stats, fetch_player_pergame_stats, 
    load_and_merge_data, calculate_player_value_metrics, calculate_team_metrics, 
    add_team_logos, fetch_historical_data, build_similarity_model, find_similar_players, 
    find_replacement_players
)
from .visualizations import (
    create_efficiency_quadrant, create_team_grid, create_salary_impact_scatter, create_underpaid_bar,
    create_overpaid_bar, create_player_beeswarm, create_team_radar_chart
)
from .config import CURRENT_SEASON
from .cache_manager import cache



def fig_to_json(fig) -> str:
    """Convert Plotly figure to JSON without binary array encoding."""
    import base64
    import struct
    
    fig_dict = fig.to_dict()
    
    DATA_FIELDS = {'x', 'y', 'z', 'r', 'theta', 'values', 'labels', 'ids', 'parents',
                   'lat', 'lon', 'locations', 'marker_color', 'marker_size', 
                   'customdata', 'hovertext', 'text', 'textposition'}
    
    def decode_bdata(dtype: str, bdata: str) -> list:
        """Decode Plotly's binary array format."""
        raw = base64.b64decode(bdata)
        
        dtype_map = {
            'f8': ('d', 8), 'f4': ('f', 4), 'i8': ('q', 8), 'i4': ('i', 4),
            'i2': ('h', 2), 'i1': ('b', 1), 'u8': ('Q', 8), 'u4': ('I', 4),
            'u2': ('H', 2), 'u1': ('B', 1)
        }
        
        if dtype in dtype_map:
            fmt, size = dtype_map[dtype]
            count = len(raw) // size
            return list(struct.unpack(f'<{count}{fmt}', raw))
        
        return [{'dtype': dtype, 'bdata': bdata}]
    
    def convert_arrays(obj, parent_key=None):
        """Recursively convert binary arrays and numpy types."""
        if isinstance(obj, dict):
            if 'dtype' in obj and 'bdata' in obj:
                if parent_key in DATA_FIELDS:
                    return decode_bdata(obj['dtype'], obj['bdata'])
                return obj
            return {k: convert_arrays(v, parent_key=k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item, parent_key=parent_key) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.generic, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return obj
    
    return json.dumps(convert_arrays(fig_dict))

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.generic, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.generic, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj
    
app = FastAPI(
    title="Sieve NBA Analytics API",
    description="REST API for NBA player value and efficiency analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_season_data_cache = {}
_similarity_model_cache = {}
_diamond_finder_cache = {}

def load_season_data(season: str):
    """Load data for a specific season with caching."""
    if season in _season_data_cache:
        df_mem, _ = _season_data_cache[season]
        if df_mem is not None and not df_mem.empty and 'PTS' in df_mem.columns:
            return _season_data_cache[season]
        else:
            print(f"API: In-memory cache for {season} is stale. Reloading...")

    print(f"API: Loading data for season {season}...")

    try:
        df_players: pd.DataFrame = cache.load_player_analysis(season=season)
        df_teams: pd.DataFrame = cache.load_team_efficiency(season=season)

        is_stale = df_players is None or df_players.empty or 'PTS' not in df_players.columns
        
        if not is_stale:
            print(f"API: Loaded from DB cache for {season}")
            if df_teams is None or df_teams.empty:
                df_teams = data_processing.calculate_team_metrics(df_players, season=season)
            df_teams = data_processing.add_team_logos(df_teams)
            _season_data_cache[season] = (df_players, df_teams)
            return df_players, df_teams

        print(f"API: Cache miss for {season}, running full pipeline...")
        lebron_file = 'data/LEBRON.csv' if season == '2024-25' else f'data/LEBRON_{season}.csv'
        df_players = data_processing.load_and_merge_data(lebron_file=lebron_file, season=season, from_db=True)
        df_players = data_processing.calculate_player_value_metrics(df_players, season=season)
        df_teams = data_processing.calculate_team_metrics(df_players, season=season)
        df_teams = data_processing.add_team_logos(df_teams)
        _season_data_cache[season] = (df_players, df_teams)
        return df_players, df_teams

    except Exception as e:
        print(f"API: Error loading data for {season}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def get_similarity_model():
    """Load or build the historical similarity model."""
    if 'model' in _similarity_model_cache:
        return _similarity_model_cache

    try:
        model_data = cache.load_similarity_model()
        if model_data:
            _similarity_model_cache.update(model_data)
            return _similarity_model_cache

        df_history: pd.DataFrame = data_processing.fetch_historical_data()
        result = data_processing.build_similarity_model(df_history)
        if result and len(result) == 4:
            knn_model, knn_scaler, df_model_data, knn_feature_info = result
            model_data = {'model': knn_model, 'scaler': knn_scaler, 'df': df_model_data, 'feature_info': knn_feature_info}
            cache.save_similarity_model(model_data)
            _similarity_model_cache.update(model_data)
            return _similarity_model_cache
    except Exception as e:
        print(f"API: Error building similarity model: {e}")

    return None

def get_diamond_finder_model(season: str):
    """Get or build Diamond Finder model for a season."""
    if season in _diamond_finder_cache:
        return _diamond_finder_cache[season]

    try:
        result = data_processing.load_and_merge_data(season=season, from_db=True)
        df = result[0] if isinstance(result, tuple) else result
        df = data_processing.calculate_player_value_metrics(df, season=season)
        model, scaler, df_filtered, feature_info = data_processing.build_current_season_similarity(df, season=season)
        if model is not None:
            _diamond_finder_cache[season] = {'model': model, 'scaler': scaler, 'df': df_filtered, 'feature_info': feature_info}
            return _diamond_finder_cache[season]
    except Exception as e:
        print(f"API: Error building diamond finder model: {e}")

    return None

def run_caching_process():
    """Runs the full data caching pipeline."""
    print("Starting Sieve data caching process...")
    seasons_to_process = get_season_list(start_year=2022)
    seasons_to_process.append(CURRENT_SEASON)
    seasons_to_process = sorted(list(set(seasons_to_process)), reverse=True)

    for season in seasons_to_process:
        print(f"\n{'='*60}\nPROCESSING SEASON: {season}\n{'='*60}\n")
        fetch_standings(force_refresh=True, season=season)
        fetch_nba_advanced_stats(force_refresh=True, season=season)
        fetch_player_pergame_stats(force_refresh=True, season=season)
        try:
            fetch_lineup_stats(force_refresh=True, season=season)
        except Exception as e:
            print(f"Could not fetch lineup stats for {season}: {e}")

        lebron_file = f'data/LEBRON_{season}.csv' if season != '2024-25' else 'data/LEBRON.csv'
        df_players: pd.DataFrame = load_and_merge_data(lebron_file=lebron_file, season=season, from_db=False)
        if df_players.empty:
            print(f"No player data found for {season}. Skipping.")
            continue

        df_players_processed = calculate_player_value_metrics(df_players, season=season)
        df_teams: pd.DataFrame = calculate_team_metrics(df_players_processed, season=season)
        if not df_teams.empty:
            df_teams_with_logos = add_team_logos(df_teams)
            cache.save_team_efficiency(df_teams_with_logos, season=season)
        print(f"Successfully processed and cached all data for {season}")

    print("\nProcessing historical data for similarity model...")
    df_history: pd.DataFrame = fetch_historical_data(force_refresh=True)
    if not df_history.empty:
        model, scaler, df, features = build_similarity_model(df_history)
        if model:
            cache.save_similarity_model({'model': model, 'scaler': scaler, 'df': df, 'feature_info': features})
            print("Saved similarity model to cache.")
    print("\nPipeline finished. All seasons processed and cached.")


def fig_to_json(fig) -> str:
    """Convert Plotly figure to JSON without binary array encoding."""
    import base64
    import struct
    
    fig_dict = fig.to_dict()
    
    DATA_FIELDS = {'x', 'y', 'z', 'r', 'theta', 'values', 'labels', 'ids', 'parents',
                   'lat', 'lon', 'locations', 'marker_color', 'marker_size', 
                   'customdata', 'hovertext', 'text', 'textposition'}
    
    def decode_bdata(dtype: str, bdata: str) -> list:
        """Decode Plotly's binary array format."""
        raw = base64.b64decode(bdata)
        
        dtype_map = {
            'f8': ('d', 8), 'f4': ('f', 4), 'i8': ('q', 8), 'i4': ('i', 4),
            'i2': ('h', 2), 'i1': ('b', 1), 'u8': ('Q', 8), 'u4': ('I', 4),
            'u2': ('H', 2), 'u1': ('B', 1)
        }
        
        if dtype in dtype_map:
            fmt, size = dtype_map[dtype]
            count = len(raw) // size
            return list(struct.unpack(f'<{count}{fmt}', raw))
        
        return [{'dtype': dtype, 'bdata': bdata}]
    
    def convert_arrays(obj, parent_key=None):
        """Recursively convert binary arrays and numpy types."""
        if isinstance(obj, dict):
            if 'dtype' in obj and 'bdata' in obj:
                if parent_key in DATA_FIELDS:
                    return decode_bdata(obj['dtype'], obj['bdata'])
                return obj
            return {k: convert_arrays(v, parent_key=k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item, parent_key=parent_key) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.generic, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return obj
    
    return json.dumps(convert_arrays(fig_dict))

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.generic, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.generic, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj

app = FastAPI(
    title="Sieve NBA Analytics API",
    description="REST API for NBA player value and efficiency analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_season_data_cache = {}
_similarity_model_cache = {}
_diamond_finder_cache = {}

def load_season_data(season: str):
    """Load data for a specific season with caching."""
    if season in _season_data_cache:
        df_mem, _ = _season_data_cache[season]
        if df_mem is not None and not df_mem.empty and 'PTS' in df_mem.columns:
            return _season_data_cache[season]
        else:
            print(f"API: In-memory cache for {season} is stale. Reloading...")

    print(f"API: Loading data for season {season}...")

    try:
        df_players: pd.DataFrame = cache.load_player_analysis(season=season)
        df_teams: pd.DataFrame = cache.load_team_efficiency(season=season)

        is_stale = df_players is None or df_players.empty or 'PTS' not in df_players.columns
        
        if not is_stale:
            print(f"API: Loaded from DB cache for {season}")
            if df_teams is None or df_teams.empty:
                df_teams = data_processing.calculate_team_metrics(df_players, season=season)
            df_teams = data_processing.add_team_logos(df_teams)
            _season_data_cache[season] = (df_players, df_teams)
            return df_players, df_teams

        print(f"API: Cache miss for {season}, running full pipeline...")
        lebron_file = 'data/LEBRON.csv' if season == '2024-25' else f'data/LEBRON_{season}.csv'
        df_players = data_processing.load_and_merge_data(lebron_file=lebron_file, season=season, from_db=True)
        df_players = data_processing.calculate_player_value_metrics(df_players, season=season)
        df_teams = data_processing.calculate_team_metrics(df_players, season=season)
        df_teams = data_processing.add_team_logos(df_teams)
        _season_data_cache[season] = (df_players, df_teams)
        return df_players, df_teams

    except Exception as e:
        print(f"API: Error loading data for {season}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def get_similarity_model():
    """Load or build the historical similarity model."""
    if 'model' in _similarity_model_cache:
        return _similarity_model_cache

    try:
        model_data = cache.load_similarity_model()
        if model_data:
            _similarity_model_cache.update(model_data)
            return _similarity_model_cache

        df_history: pd.DataFrame = data_processing.fetch_historical_data()
        result = data_processing.build_similarity_model(df_history)
        if result and len(result) == 4:
            knn_model, knn_scaler, df_model_data, knn_feature_info = result
            model_data = {'model': knn_model, 'scaler': knn_scaler, 'df': df_model_data, 'feature_info': knn_feature_info}
            cache.save_similarity_model(model_data)
            _similarity_model_cache.update(model_data)
            return _similarity_model_cache
    except Exception as e:
        print(f"API: Error building similarity model: {e}")

    return None

def get_diamond_finder_model(season: str):
    """Get or build Diamond Finder model for a season."""
    if season in _diamond_finder_cache:
        return _diamond_finder_cache[season]

    try:
        result = data_processing.load_and_merge_data(season=season, from_db=True)
        df = result[0] if isinstance(result, tuple) else result
        df = data_processing.calculate_player_value_metrics(df, season=season)
        model, scaler, df_filtered, feature_info = data_processing.build_current_season_similarity(df, season=season)
        if model is not None:
            _diamond_finder_cache[season] = {'model': model, 'scaler': scaler, 'df': df_filtered, 'feature_info': feature_info}
            return _diamond_finder_cache[season]
    except Exception as e:
        print(f"API: Error building diamond finder model: {e}")

    return None

def run_caching_process():
    """Runs the full data caching pipeline."""
    print("Starting Sieve data caching process...")
    seasons_to_process = get_season_list(start_year=2022)
    seasons_to_process.append(CURRENT_SEASON)
    seasons_to_process = sorted(list(set(seasons_to_process)), reverse=True)

    for season in seasons_to_process:
        print(f"\n{'='*60}\nPROCESSING SEASON: {season}\n{'='*60}\n")
        fetch_standings(force_refresh=True, season=season)
        fetch_nba_advanced_stats(force_refresh=True, season=season)
        fetch_player_pergame_stats(force_refresh=True, season=season)
        try:
            data_processing.fetch_lineup_stats(force_refresh=True, season=season)
        except Exception as e:
            print(f"Could not fetch lineup stats for {season}: {e}")

        lebron_file = f'data/LEBRON_{season}.csv' if season != '2024-25' else 'data/LEBRON.csv'
        df_players: pd.DataFrame = load_and_merge_data(lebron_file=lebron_file, season=season, from_db=False)
        if df_players.empty:
            print(f"No player data found for {season}. Skipping.")
            continue

        df_players_processed = calculate_player_value_metrics(df_players, season=season)
        df_teams: pd.DataFrame = calculate_team_metrics(df_players_processed, season=season)
        if not df_teams.empty:
            df_teams_with_logos = add_team_logos(df_teams)
            cache.save_team_efficiency(df_teams_with_logos, season=season)
        print(f"Successfully processed and cached all data for {season}")

    print("\nProcessing historical data for similarity model...")
    df_history: pd.DataFrame = fetch_historical_data(force_refresh=True)
    if not df_history.empty:
        model, scaler, df, features = build_similarity_model(df_history)
        if model:
            cache.save_similarity_model({'model': model, 'scaler': scaler, 'df': df, 'feature_info': features})
            print("Saved similarity model to cache.")
    print("\nPipeline finished. All seasons processed and cached.")


def fig_to_json(fig) -> str:
    """Convert Plotly figure to JSON without binary array encoding."""
    import base64
    import struct
    
    fig_dict = fig.to_dict()
    
    DATA_FIELDS = {'x', 'y', 'z', 'r', 'theta', 'values', 'labels', 'ids', 'parents',
                   'lat', 'lon', 'locations', 'marker_color', 'marker_size', 
                   'customdata', 'hovertext', 'text', 'textposition'}
    
    def decode_bdata(dtype: str, bdata: str) -> list:
        """Decode Plotly's binary array format."""
        raw = base64.b64decode(bdata)
        
        dtype_map = {
            'f8': ('d', 8), 'f4': ('f', 4), 'i8': ('q', 8), 'i4': ('i', 4),
            'i2': ('h', 2), 'i1': ('b', 1), 'u8': ('Q', 8), 'u4': ('I', 4),
            'u2': ('H', 2), 'u1': ('B', 1)
        }
        
        if dtype in dtype_map:
            fmt, size = dtype_map[dtype]
            count = len(raw) // size
            return list(struct.unpack(f'<{count}{fmt}', raw))
        
        return [{'dtype': dtype, 'bdata': bdata}]
    
    def convert_arrays(obj, parent_key=None):
        """Recursively convert binary arrays and numpy types."""
        if isinstance(obj, dict):
            if 'dtype' in obj and 'bdata' in obj:
                if parent_key in DATA_FIELDS:
                    return decode_bdata(obj['dtype'], obj['bdata'])
                return obj
            return {k: convert_arrays(v, parent_key=k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item, parent_key=parent_key) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.generic, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return obj
    
    return json.dumps(convert_arrays(fig_dict))

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.generic, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.generic, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj

app = FastAPI(
    title="Sieve NBA Analytics API",
    description="REST API for NBA player value and efficiency analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_season_data_cache = {}
_similarity_model_cache = {}
_diamond_finder_cache = {}

def load_season_data(season: str):
    """Load data for a specific season with caching."""
    if season in _season_data_cache:
        df_mem, _ = _season_data_cache[season]
        if df_mem is not None and not df_mem.empty and 'PTS' in df_mem.columns:
            return _season_data_cache[season]
        else:
            print(f"API: In-memory cache for {season} is stale. Reloading...")

    print(f"API: Loading data for season {season}...")

    try:
        df_players = cache.load_player_analysis(season=season)
        df_teams = cache.load_team_efficiency(season=season)

        is_stale = df_players is None or df_players.empty or 'PTS' not in df_players.columns
        
        if not is_stale:
            print(f"API: Loaded from DB cache for {season}")
            if df_teams is None or df_teams.empty:
                df_teams = data_processing.calculate_team_metrics(df_players, season=season)
            df_teams = data_processing.add_team_logos(df_teams)
            _season_data_cache[season] = (df_players, df_teams)
            return df_players, df_teams

        print(f"API: Cache miss for {season}, running full pipeline...")
        lebron_file = 'data/LEBRON.csv' if season == '2024-25' else f'data/LEBRON_{season}.csv'
        df_players = data_processing.load_and_merge_data(lebron_file=lebron_file, season=season, from_db=True)
        df_players = data_processing.calculate_player_value_metrics(df_players, season=season)
        df_teams = data_processing.calculate_team_metrics(df_players, season=season)
        df_teams = data_processing.add_team_logos(df_teams)
        _season_data_cache[season] = (df_players, df_teams)
        return df_players, df_teams

    except Exception as e:
        print(f"API: Error loading data for {season}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def get_similarity_model():
    """Load or build the historical similarity model."""
    if 'model' in _similarity_model_cache:
        return _similarity_model_cache

    try:
        model_data = cache.load_similarity_model()
        if model_data:
            _similarity_model_cache.update(model_data)
            return _similarity_model_cache

        df_history = data_processing.fetch_historical_data()
        result = data_processing.build_similarity_model(df_history)
        if result and len(result) == 4:
            knn_model, knn_scaler, df_model_data, knn_feature_info = result
            model_data = {'model': knn_model, 'scaler': knn_scaler, 'df': df_model_data, 'feature_info': knn_feature_info}
            cache.save_similarity_model(model_data)
            _similarity_model_cache.update(model_data)
            return _similarity_model_cache
    except Exception as e:
        print(f"API: Error building similarity model: {e}")

    return None

def get_diamond_finder_model(season: str):
    """Get or build Diamond Finder model for a season."""
    if season in _diamond_finder_cache:
        return _diamond_finder_cache[season]

    try:
        result = data_processing.load_and_merge_data(season=season, from_db=True)
        df = result[0] if isinstance(result, tuple) else result
        df = data_processing.calculate_player_value_metrics(df, season=season)
        model, scaler, df_filtered, feature_info = data_processing.build_current_season_similarity(df, season=season)
        if model is not None:
            _diamond_finder_cache[season] = {'model': model, 'scaler': scaler, 'df': df_filtered, 'feature_info': feature_info}
            return _diamond_finder_cache[season]
    except Exception as e:
        print(f"API: Error building diamond finder model: {e}")

    return None

def run_caching_process():
    """Runs the full data caching pipeline."""
    print("Starting Sieve data caching process...")
    seasons_to_process = get_season_list(start_year=2022)
    seasons_to_process.append(CURRENT_SEASON)
    seasons_to_process = sorted(list(set(seasons_to_process)), reverse=True)

    for season in seasons_to_process:
        print(f"\n{'='*60}\nPROCESSING SEASON: {season}\n{'='*60}\n")
        fetch_standings(force_refresh=True, season=season)
        fetch_nba_advanced_stats(force_refresh=True, season=season)
        fetch_player_pergame_stats(force_refresh=True, season=season)
        try:
            fetch_lineup_stats(force_refresh=True, season=season)
        except Exception as e:
            print(f"Could not fetch lineup stats for {season}: {e}")

        lebron_file = f'data/LEBRON_{season}.csv' if season != '2024-25' else 'data/LEBRON.csv'
        df_players = load_and_merge_data(lebron_file=lebron_file, season=season, from_db=False)
        if df_players.empty:
            print(f"No player data found for {season}. Skipping.")
            continue

        df_players_processed = calculate_player_value_metrics(df_players, season=season)
        df_teams = calculate_team_metrics(df_players_processed, season=season)
        if not df_teams.empty:
            df_teams_with_logos = add_team_logos(df_teams)
            cache.save_team_efficiency(df_teams_with_logos, season=season)
        print(f"Successfully processed and cached all data for {season}")

    print("\nProcessing historical data for similarity model...")
    df_history = fetch_historical_data(force_refresh=True)
    if not df_history.empty:
        model, scaler, df, features = build_similarity_model(df_history)
        if model:
            cache.save_similarity_model({'model': model, 'scaler': scaler, 'df': df, 'feature_info': features})
            print("Saved similarity model to cache.")
    print("\nPipeline finished. All seasons processed and cached.")





def fig_to_json(fig) -> str:
    """Convert Plotly figure to JSON without binary array encoding.
    
    When using pandas Series directly in Plotly, it encodes arrays as binary (bdata).
    This function decodes the binary arrays back to regular lists.
    Only decodes binary data for known data fields (x, y, z, text, etc.) to avoid
    corrupting other internal Plotly structures.
    """
    import base64
    import struct
    
    fig_dict = fig.to_dict()
    
    # Fields that contain actual data arrays and should be decoded from binary
    DATA_FIELDS = {'x', 'y', 'z', 'r', 'theta', 'values', 'labels', 'ids', 'parents',
                   'lat', 'lon', 'locations', 'marker_color', 'marker_size', 
                   'customdata', 'hovertext', 'text', 'textposition'}
    
    def decode_bdata(dtype: str, bdata: str) -> list:
        """Decode Plotly's binary array format."""
        raw = base64.b64decode(bdata)
        
        dtype_map = {
            'f8': ('d', 8),  # float64
            'f4': ('f', 4),  # float32
            'i8': ('q', 8),  # int64
            'i4': ('i', 4),  # int32
            'i2': ('h', 2),  # int16
            'i1': ('b', 1),  # int8
            'u8': ('Q', 8),  # uint64
            'u4': ('I', 4),  # uint32
            'u2': ('H', 2),  # uint16
            'u1': ('B', 1),  # uint8
        }
        
        if dtype in dtype_map:
            fmt, size = dtype_map[dtype]
            count = len(raw) // size
            values = list(struct.unpack(f'<{count}{fmt}', raw))
            return values
        
        # Unknown dtype - return as is (don't decode)
        return [{'dtype': dtype, 'bdata': bdata}]
    
    def convert_arrays(obj, parent_key=None):
        """Recursively convert binary arrays and numpy types."""
        if isinstance(obj, dict):
            # Check for binary data format - only decode for data fields
            if 'dtype' in obj and 'bdata' in obj:
                # Only decode if this is a known data field
                if parent_key in DATA_FIELDS:
                    return decode_bdata(obj['dtype'], obj['bdata'])
                # Otherwise keep as-is (will be serialized as dict)
                return obj
            return {k: convert_arrays(v, parent_key=k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item, parent_key=parent_key) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.generic, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj) if hasattr(pd, 'isna') else False:
            return None
        return obj
    
    return json.dumps(convert_arrays(fig_dict))
 
 
def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.generic, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.generic, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj





def fig_to_json(fig) -> str:
    """Convert Plotly figure to JSON without binary array encoding.
    
    When using pandas Series directly in Plotly, it encodes arrays as binary (bdata).
    This function decodes the binary arrays back to regular lists.
    Only decodes binary data for known data fields (x, y, z, text, etc.) to avoid
    corrupting other internal Plotly structures.
    """
    import base64
    import struct
    
    fig_dict = fig.to_dict()
    
    # Fields that contain actual data arrays and should be decoded from binary
    DATA_FIELDS = {'x', 'y', 'z', 'r', 'theta', 'values', 'labels', 'ids', 'parents',
                   'lat', 'lon', 'locations', 'marker_color', 'marker_size', 
                   'customdata', 'hovertext', 'text', 'textposition'}
    
    def decode_bdata(dtype: str, bdata: str) -> list:
        """Decode Plotly's binary array format."""
        raw = base64.b64decode(bdata)
        
        dtype_map = {
            'f8': ('d', 8),  # float64
            'f4': ('f', 4),  # float32
            'i8': ('q', 8),  # int64
            'i4': ('i', 4),  # int32
            'i2': ('h', 2),  # int16
            'i1': ('b', 1),  # int8
            'u8': ('Q', 8),  # uint64
            'u4': ('I', 4),  # uint32
            'u2': ('H', 2),  # uint16
            'u1': ('B', 1),  # uint8
        }
        
        if dtype in dtype_map:
            fmt, size = dtype_map[dtype]
            count = len(raw) // size
            values = list(struct.unpack(f'<{count}{fmt}', raw))
            return values
        
        # Unknown dtype - return as is (don't decode)
        return [{'dtype': dtype, 'bdata': bdata}]
    
    def convert_arrays(obj, parent_key=None):
        """Recursively convert binary arrays and numpy types."""
        if isinstance(obj, dict):
            # Check for binary data format - only decode for data fields
            if 'dtype' in obj and 'bdata' in obj:
                # Only decode if this is a known data field
                if parent_key in DATA_FIELDS:
                    return decode_bdata(obj['dtype'], obj['bdata'])
                # Otherwise keep as-is (will be serialized as dict)
                return obj
            return {k: convert_arrays(v, parent_key=k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item, parent_key=parent_key) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.generic, np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj) if hasattr(pd, 'isna') else False:
            return None
        return obj
    
    return json.dumps(convert_arrays(fig_dict))
 
 
def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.generic, np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.generic, np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj







def fig_to_json(fig) -> str:
    """Convert Plotly figure to JSON without binary array encoding.
    
    When using pandas Series directly in Plotly, it encodes arrays as binary (bdata).
    This function decodes the binary arrays back to regular lists.
    Only decodes binary data for known data fields (x, y, z, text, etc.) to avoid
    corrupting other internal Plotly structures.
    """
    import base64
    import struct
    
    fig_dict = fig.to_dict()
    
    # Fields that contain actual data arrays and should be decoded from binary
    DATA_FIELDS = {'x', 'y', 'z', 'r', 'theta', 'values', 'labels', 'ids', 'parents',
                   'lat', 'lon', 'locations', 'marker_color', 'marker_size', 
                   'customdata', 'hovertext', 'text', 'textposition'}
    
    def decode_bdata(dtype: str, bdata: str) -> list:
        """Decode Plotly's binary array format."""
        raw = base64.b64decode(bdata)
        
        dtype_map = {
            'f8': ('d', 8),  # float64
            'f4': ('f', 4),  # float32
            'i8': ('q', 8),  # int64
            'i4': ('i', 4),  # int32
            'i2': ('h', 2),  # int16
            'i1': ('b', 1),  # int8
            'u8': ('Q', 8),  # uint64
            'u4': ('I', 4),  # uint32
            'u2': ('H', 2),  # uint16
            'u1': ('B', 1),  # uint8
        }
        
        if dtype in dtype_map:
            fmt, size = dtype_map[dtype]
            count = len(raw) // size
            values = list(struct.unpack(f'<{count}{fmt}', raw))
            return values
        
        # Unknown dtype - return as is (don't decode)
        return [{'dtype': dtype, 'bdata': bdata}]
    
    def convert_arrays(obj, parent_key=None):
        """Recursively convert binary arrays and numpy types."""
        if isinstance(obj, dict):
            # Check for binary data format - only decode for data fields
            if 'dtype' in obj and 'bdata' in obj:
                # Only decode if this is a known data field
                if parent_key in DATA_FIELDS:
                    return decode_bdata(obj['dtype'], obj['bdata'])
                # Otherwise keep as-is (will be serialized as dict)
                return obj
            return {k: convert_arrays(v, parent_key=k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item, parent_key=parent_key) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.generic, np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj) if hasattr(pd, 'isna') else False:
            return None
        return obj
    
    return json.dumps(convert_arrays(fig_dict))
 
 
def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.generic, np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.generic, np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj
from . import visualizations
from .config import CURRENT_SEASON
from .cache_manager import cache




def fig_to_json(fig) -> str:
    """Convert Plotly figure to JSON without binary array encoding.
    
    When using pandas Series directly in Plotly, it encodes arrays as binary (bdata).
    This function decodes the binary arrays back to regular lists.
    Only decodes binary data for known data fields (x, y, z, text, etc.) to avoid
    corrupting other internal Plotly structures.
    """
    import base64
    import struct
    
    fig_dict = fig.to_dict()
    
    # Fields that contain actual data arrays and should be decoded from binary
    DATA_FIELDS = {'x', 'y', 'z', 'r', 'theta', 'values', 'labels', 'ids', 'parents',
                   'lat', 'lon', 'locations', 'marker_color', 'marker_size', 
                   'customdata', 'hovertext', 'text', 'textposition'}
    
    def decode_bdata(dtype: str, bdata: str) -> list:
        """Decode Plotly's binary array format."""
        raw = base64.b64decode(bdata)
        
        dtype_map = {
            'f8': ('d', 8),  # float64
            'f4': ('f', 4),  # float32
            'i8': ('q', 8),  # int64
            'i4': ('i', 4),  # int32
            'i2': ('h', 2),  # int16
            'i1': ('b', 1),  # int8
            'u8': ('Q', 8),  # uint64
            'u4': ('I', 4),  # uint32
            'u2': ('H', 2),  # uint16
            'u1': ('B', 1),  # uint8
        }
        
        if dtype in dtype_map:
            fmt, size = dtype_map[dtype]
            count = len(raw) // size
            values = list(struct.unpack(f'<{count}{fmt}', raw))
            return values
        
        # Unknown dtype - return as is (don't decode)
        return {'dtype': dtype, 'bdata': bdata}
    
    def convert_arrays(obj, parent_key=None):
        """Recursively convert binary arrays and numpy types."""
        if isinstance(obj, dict):
            # Check for binary data format - only decode for data fields
            if 'dtype' in obj and 'bdata' in obj:
                # Only decode if this is a known data field
                if parent_key in DATA_FIELDS:
                    return decode_bdata(obj['dtype'], obj['bdata'])
                # Otherwise keep as-is (will be serialized as dict)
                return obj
            return {k: convert_arrays(v, parent_key=k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item, parent_key=parent_key) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj) if hasattr(pd, 'isna') else False:
            return None
        return obj
    
    return json.dumps(convert_arrays(fig_dict))
 
 
def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.generic, np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.generic, np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj
from . import visualizations
from .config import CURRENT_SEASON
from .cache_manager import cache

# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Sieve NBA Analytics API",
    description="REST API for NBA player value and efficiency analysis",
    version="1.0.0"
)

# CORS middleware - allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA CACHING (reuses logic from dashboard.py)
# =============================================================================

_season_data_cache = {}
_similarity_model_cache = {}
_diamond_finder_cache = {}


def load_season_data(season: str):
    """Load data for a specific season with caching."""
    if season in _season_data_cache:
        # Also check in-memory cache for missing columns
        df_mem, _ = _season_data_cache[season]
        required_cols = ['PTS', 'REB', 'AST']
        if df_mem is not None and not df_mem.empty and all(col in df_mem.columns for col in required_cols):
            return _season_data_cache[season]
        else:
             print(f"API: In-memory cache for {season} is stale. Reloading...")

    print(f"API: Loading data for season {season}...")

    try:
        # Try DB cache first
        df_players = cache.load_player_analysis(season=season)
        df_teams = cache.load_team_efficiency(season=season)

        # Check if DB cache has the new stat columns
        required_cols = ['PTS', 'REB', 'AST']
        is_stale = False
        if df_players is not None and not df_players.empty:
             miss_cols = [c for c in required_cols if c not in df_players.columns]
             if miss_cols:
                 print(f"API: DB cache for {season} is stale (missing {miss_cols}). Reloading...")
                 is_stale = True

        if df_players is not None and not df_players.empty and not is_stale:
            print(f"API: Loaded from DB cache for {season}")

            if df_teams is None or df_teams.empty:
                df_teams = data_processing.calculate_team_metrics(df_players, season=season)

            df_teams = data_processing.add_team_logos(df_teams)
            _season_data_cache[season] = (df_players, df_teams)
            return df_players, df_teams

        # Fallback: full pipeline
        print(f"API: Cache miss for {season}, running full pipeline...")

        lebron_file = 'data/LEBRON.csv' if season == '2024-25' else f'data/LEBRON_{season}.csv'

        df_players = data_processing.load_and_merge_data(
            lebron_file=lebron_file,
            season=season,
            from_db=True
        )
        df_players = data_processing.calculate_player_value_metrics(df_players, season=season)
        df_teams = data_processing.calculate_team_metrics(df_players, season=season)
        df_teams = data_processing.add_team_logos(df_teams)

        _season_data_cache[season] = (df_players, df_teams)
        return df_players, df_teams

    except Exception as e:
        print(f"API: Error loading data for {season}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()


def get_similarity_model():
    """Load or build the historical similarity model."""
    if 'model' in _similarity_model_cache:
        return _similarity_model_cache

    try:
        df_history = data_processing.fetch_historical_data()
        result = data_processing.build_similarity_model(df_history)

        if result and len(result) == 4:
            knn_model, knn_scaler, df_model_data, knn_feature_info = result
            _similarity_model_cache['model'] = knn_model
            _similarity_model_cache['scaler'] = knn_scaler
            _similarity_model_cache['df'] = df_model_data
            _similarity_model_cache['feature_info'] = knn_feature_info
            return _similarity_model_cache
    except Exception as e:
        print(f"API: Error building similarity model: {e}")

    return None


def get_diamond_finder_model(season: str):
    """Get or build Diamond Finder model for a season."""
    if season in _diamond_finder_cache:
        return _diamond_finder_cache[season]

    try:
        result = data_processing.load_and_merge_data(season=season, from_db=True)
        df = result[0] if isinstance(result, tuple) else result
        df = data_processing.calculate_player_value_metrics(df, season=season)

        model, scaler, df_filtered, feature_info = data_processing.build_current_season_similarity(df, season=season)

        if model is not None:
            _diamond_finder_cache[season] = {
                'model': model,
                'scaler': scaler,
                'df': df_filtered,
                'feature_info': feature_info
            }
            return _diamond_finder_cache[season]
    except Exception as e:
        print(f"API: Error building diamond finder model: {e}")

    return None


# =============================================================================
# API ENDPOINTS - DATA
# =============================================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/logo/{team_id}")
def get_team_logo(team_id: int):
    """Proxy endpoint to serve NBA team logos (bypasses CORS restrictions)."""
    import requests
    
    url = f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return Response(
                content=resp.content,
                media_type="image/svg+xml",
                headers={"Cache-Control": "public, max-age=86400"}
            )
    except Exception:
        pass
    
    placeholder = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="#333"/></svg>'
    return Response(content=placeholder, media_type="image/svg+xml")


@app.get("/api/seasons")
def get_seasons():
    """Get list of available seasons."""
    try:
        seasons = cache.list_lebron_seasons()
        if not seasons:
            seasons = [CURRENT_SEASON]
        return {"seasons": sorted(seasons, reverse=True), "current": CURRENT_SEASON}
    except Exception as e:
        return {"seasons": [CURRENT_SEASON], "current": CURRENT_SEASON}


@app.get("/api/players")
def get_players(
    season: str = Query(default=CURRENT_SEASON, description="NBA season (e.g., '2024-25')"),
    min_lebron: Optional[float] = Query(default=None, description="Minimum LEBRON score"),
    max_lebron: Optional[float] = Query(default=None, description="Maximum LEBRON score"),
    min_salary: Optional[float] = Query(default=None, description="Minimum salary in millions"),
    max_salary: Optional[float] = Query(default=None, description="Maximum salary in millions"),
    search: Optional[str] = Query(default=None, description="Player name search")
):
    """Get player data with optional filters."""
    df, _ = load_season_data(season)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for season {season}")

    # Apply filters
    if min_lebron is not None:
        df = df[df['LEBRON'] >= min_lebron]

    if max_lebron is not None:
        df = df[df['LEBRON'] <= max_lebron]

    if min_salary is not None:
        df = df[df['current_year_salary'] >= min_salary * 1_000_000]

    if max_salary is not None:
        df = df[df['current_year_salary'] <= max_salary * 1_000_000]

    if search:
        df = df[df['player_name'].str.contains(search, case=False, na=False)]

    # Convert to records, handling NaN
    records = df.replace({pd.NA: None, float('nan'): None}).to_dict(orient='records')
    return {"players": records, "count": len(records), "season": season}


@app.get("/api/teams")
def get_teams(season: str = Query(default=CURRENT_SEASON)):
    """Get team data."""
    _, df_teams = load_season_data(season)

    if df_teams.empty:
        raise HTTPException(status_code=404, detail=f"No team data for season {season}")

    records = df_teams.replace({pd.NA: None, float('nan'): None}).to_dict(orient='records')
    return {"teams": records, "count": len(records), "season": season}


@app.get("/api/overview")
def get_overview(season: str = Query(default=CURRENT_SEASON)):
    """Get overview statistics for the landing page."""
    df, df_teams = load_season_data(season)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for season {season}")

    # Helper to build rich player object
    def build_player_dict(row):
        d = {
            "name": row.get('player_name', 'Unknown'),
            "team": row.get('Team(s)', row.get('Team', 'N/A')),
            "value_gap": float(row.get('value_gap', 0)),
            "lebron": float(row.get('LEBRON', 0)),
            "o_lebron": float(row.get('O-LEBRON', 0)) if pd.notna(row.get('O-LEBRON')) else None,
            "d_lebron": float(row.get('D-LEBRON', 0)) if pd.notna(row.get('D-LEBRON')) else None,
            "salary": float(row.get('current_year_salary', 0)) / 1_000_000,
            "archetype": row.get('archetype', row.get('Offensive Archetype', '')),
            "role": row.get('Rotation Role', ''),
            
            # Per-Game Stats
            "ppg": row.get('PTS'),
            "rpg": row.get('REB'),
            "apg": row.get('AST'),
            "spg": row.get('STL'),
            "bpg": row.get('BLK'),
            
            # Shooting
            "fg_pct": row.get('FG_PCT'),
            "three_pct": row.get('FG3_PCT'),
            "ft_pct": row.get('FT_PCT'),
            "ts_pct": row.get('TS_PCT'),
            
            # Percentiles
            "ppg_pct": row.get('PTS_PCT'),
            "rpg_pct": row.get('REB_PCT'),
            "apg_pct": row.get('AST_PCT'),
            "spg_pct": row.get('STL_PCT'),
            "bpg_pct": row.get('BLK_PCT'),
            "fg_pct_pct": row.get('FG_PCT_PCT'),
            "three_pct_pct": row.get('FG3_PCT_PCT'),
            "ts_pct_pct": row.get('TS_PCT_PCT'),
        }
        if 'PLAYER_ID' in row and pd.notna(row['PLAYER_ID']):
            d['player_id'] = int(row['PLAYER_ID'])
        
        # Safe float conversion for stats
        for k in ['ppg', 'rpg', 'apg', 'spg', 'bpg']:
            if d[k] is not None: d[k] = float(d[k])
            
        return d

    # Calculate stats
    num_players = len(df)
    num_teams = len(df_teams) if df_teams is not None else 0

    # Exclude estimated salaries (players with *)
    if 'current_year_salary' in df.columns and 'player_name' in df.columns:
        df_with_contracts = df[~df['player_name'].str.endswith('*', na=False)]
        avg_salary = df_with_contracts['current_year_salary'].mean() / 1_000_000 if len(df_with_contracts) > 0 else 0
    else:
        avg_salary = 0

    avg_lebron = df['LEBRON'].mean() if 'LEBRON' in df.columns else 0
    total_payroll = df_teams['Total_Payroll'].sum() / 1_000_000_000 if df_teams is not None and 'Total_Payroll' in df_teams.columns else 0

    # Top performers
    top_value_player = ""
    top_value_gap = 0
    if 'value_gap' in df.columns and not df.empty:
        top_idx = df['value_gap'].idxmax()
        top_value_player = df.loc[top_idx, 'player_name'] if 'player_name' in df.columns else "N/A"
        top_value_gap = float(df.loc[top_idx, 'value_gap'])

    most_efficient_team = ""
    if df_teams is not None and 'Efficiency_Index' in df_teams.columns and not df_teams.empty:
        top_team_idx = df_teams['Efficiency_Index'].idxmax()
        most_efficient_team = df_teams.loc[top_team_idx, 'Abbrev'] if 'Abbrev' in df_teams.columns else "N/A"

    # Top 6 underpaid players
    top_value_players = []
    if 'value_gap' in df.columns and len(df) > 0:
        underpaid = df[df['value_gap'] > 0].nlargest(6, 'value_gap')
        top_value_players = [build_player_dict(row) for _, row in underpaid.iterrows()]

    # Top 6 overpaid players
    worst_value_players = []
    if 'value_gap' in df.columns and len(df) > 0:
        overpaid = df[df['value_gap'] < 0].nsmallest(6, 'value_gap')
        worst_value_players = [build_player_dict(row) for _, row in overpaid.iterrows()]

    # Top 6 highest LEBRON performers
    top_performers = []
    if 'LEBRON' in df.columns and len(df) > 0:
        top_lebron = df.nlargest(6, 'LEBRON')
        top_performers = [build_player_dict(row) for _, row in top_lebron.iterrows()]

    return {
        "season": season,
        "num_players": num_players,
        "num_teams": num_teams,
        "avg_salary_millions": round(avg_salary, 2),
        "avg_lebron": round(avg_lebron, 3),
        "total_payroll_billions": round(total_payroll, 2),
        "top_value_player": top_value_player,
        "top_value_gap": round(top_value_gap, 2),
        "most_efficient_team": most_efficient_team,
        "top_value_players": top_value_players,
        "worst_value_players": worst_value_players,
        "top_performers": top_performers,
    }


# =============================================================================
# API ENDPOINTS - CHARTS (Return Plotly JSON)
# =============================================================================

@app.get("/api/charts/quadrant")
def get_quadrant_chart(season: str = Query(default=CURRENT_SEASON)):
    """Get efficiency quadrant chart as Plotly JSON."""
    _, df_teams = load_season_data(season)

    if df_teams.empty:
        raise HTTPException(status_code=404, detail=f"No team data for season {season}")

    fig = visualizations.create_efficiency_quadrant(df_teams)
    return Response(content=fig_to_json(fig), media_type="application/json")


@app.get("/api/charts/team-grid")
def get_team_grid_chart(season: str = Query(default=CURRENT_SEASON)):
    """Get team grid chart as Plotly JSON."""
    _, df_teams = load_season_data(season)

    if df_teams.empty:
        raise HTTPException(status_code=404, detail=f"No team data for season {season}")

    fig = visualizations.create_team_grid(df_teams)
    return Response(content=fig_to_json(fig), media_type="application/json")


@app.get("/api/charts/salary-impact")
def get_salary_impact_chart(
    season: str = Query(default=CURRENT_SEASON),
    min_lebron: float = Query(default=-5.0),
    max_lebron: float = Query(default=10.0),
    min_salary: float = Query(default=0),
    max_salary: float = Query(default=60)
):
    """Get salary vs impact scatter chart."""
    df, _ = load_season_data(season)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for season {season}")

    # Filter
    filtered_df = df[
        (df['LEBRON'] >= min_lebron) &
        (df['LEBRON'] <= max_lebron) &
        (df['current_year_salary'] >= min_salary * 1_000_000) &
        (df['current_year_salary'] <= max_salary * 1_000_000)
    ].copy()

    # Recalculate value metrics for filtered subset
    if 'current_year_salary' in filtered_df.columns and 'LEBRON' in filtered_df.columns:
        valid_salary = filtered_df['current_year_salary'].dropna()
        valid_lebron = filtered_df['LEBRON'].dropna()

        if len(valid_salary) > 0 and len(valid_lebron) > 0:
            salary_min, salary_max = valid_salary.min(), valid_salary.max()
            filtered_df['salary_norm'] = 100 * (filtered_df['current_year_salary'] - salary_min) / (salary_max - salary_min)

            lebron_min, lebron_max = valid_lebron.min(), valid_lebron.max()
            filtered_df['impact_norm'] = 100 * (filtered_df['LEBRON'] - lebron_min) / (lebron_max - lebron_min)

            filtered_df['value_gap'] = filtered_df['impact_norm'] * 1.4 - filtered_df['salary_norm'] * 0.9 - 10

    fig = visualizations.create_salary_impact_scatter(filtered_df)
    return Response(content=fig_to_json(fig), media_type="application/json")


@app.get("/api/charts/underpaid")
def get_underpaid_chart(
    season: str = Query(default=CURRENT_SEASON),
    min_lebron: float = Query(default=-5.0),
    max_lebron: float = Query(default=10.0),
    min_salary: float = Query(default=0),
    max_salary: float = Query(default=60)
):
    """Get underpaid players bar chart."""
    df, _ = load_season_data(season)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for season {season}")

    filtered_df = df[
        (df['LEBRON'] >= min_lebron) &
        (df['LEBRON'] <= max_lebron) &
        (df['current_year_salary'] >= min_salary * 1_000_000) &
        (df['current_year_salary'] <= max_salary * 1_000_000)
    ].copy()

    fig = visualizations.create_underpaid_bar(filtered_df)
    return Response(content=fig_to_json(fig), media_type="application/json")


@app.get("/api/charts/overpaid")
def get_overpaid_chart(
    season: str = Query(default=CURRENT_SEASON),
    min_lebron: float = Query(default=-5.0),
    max_lebron: float = Query(default=10.0),
    min_salary: float = Query(default=0),
    max_salary: float = Query(default=60)
):
    """Get overpaid players bar chart."""
    df, _ = load_season_data(season)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for season {season}")

    filtered_df = df[
        (df['LEBRON'] >= min_lebron) &
        (df['LEBRON'] <= max_lebron) &
        (df['current_year_salary'] >= min_salary * 1_000_000) &
        (df['current_year_salary'] <= max_salary * 1_000_000)
    ].copy()

    fig = visualizations.create_overpaid_bar(filtered_df)
    return Response(content=fig_to_json(fig), media_type="application/json")


@app.get("/api/charts/beeswarm")
def get_beeswarm_chart(
    season: str = Query(default=CURRENT_SEASON),
    min_lebron: float = Query(default=-5.0),
    max_lebron: float = Query(default=10.0),
    min_salary: float = Query(default=0),
    max_salary: float = Query(default=60)
):
    """Get player beeswarm chart."""
    df, _ = load_season_data(season)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for season {season}")

    filtered_df = df[
        (df['LEBRON'] >= min_lebron) &
        (df['LEBRON'] <= max_lebron) &
        (df['current_year_salary'] >= min_salary * 1_000_000) &
        (df['current_year_salary'] <= max_salary * 1_000_000)
    ].copy()

    fig = visualizations.create_player_beeswarm(filtered_df)
    return Response(content=fig_to_json(fig), media_type="application/json")


@app.get("/api/charts/team-radar")
def get_team_radar_chart(
    team1: str = Query(..., description="First team abbreviation"),
    team2: str = Query(..., description="Second team abbreviation")
):
    """Get team comparison radar chart."""
    radar_data_1 = data_processing.get_team_radar_data(team1)
    radar_data_2 = data_processing.get_team_radar_data(team2)
    fig = visualizations.create_team_radar_chart(radar_data_1, radar_data_2, team1, team2)
    return Response(content=fig_to_json(fig), media_type="application/json")


# =============================================================================
# API ENDPOINTS - SIMILARITY ENGINE
# =============================================================================

@app.get("/api/similarity/players")
def get_similarity_players():
    """Get list of players available for similarity search."""
    model_data = get_similarity_model()
    if model_data is None:
        return {"players": []}

    df = model_data['df']
    players = sorted(df['PLAYER_NAME'].unique().tolist())
    return {"players": players}


@app.get("/api/similarity/seasons/{player_name}")
def get_player_seasons(player_name: str):
    """Get available seasons for a player."""
    model_data = get_similarity_model()
    if model_data is None:
        return {"seasons": []}

    df = model_data['df']
    player_seasons = df[df['PLAYER_NAME'] == player_name]['SEASON_ID'].unique().tolist()
    return {"seasons": sorted(player_seasons, reverse=True)}


@app.get("/api/similarity/find")
def find_similar_players_endpoint(
    player: str = Query(..., description="Player name"),
    season: str = Query(..., description="Season ID"),
    exclude_self: bool = Query(default=True, description="Exclude same player from results")
):
    """Find similar players."""
    model_data = get_similarity_model()
    if model_data is None:
        raise HTTPException(status_code=500, detail="Similarity model not available")

    try:
        results = data_processing.find_similar_players(
            player, season, model_data['df'], model_data['model'],
            model_data['scaler'], feature_info=model_data['feature_info'],
            exclude_self=exclude_self
        )

        if not results:
            return {"similar": [], "target": None}

        # Convert numpy types to native Python types
        results = convert_numpy_types(results)

        # Get target player info
        df = model_data['df']
        target_rows = df[(df['PLAYER_NAME'] == player) & (df['SEASON_ID'] == season)]
        target_info = None
        if not target_rows.empty:
            row = target_rows.iloc[0]
            target_info = {
                "name": player,
                "season": season,
                "player_id": int(row['PLAYER_ID']) if pd.notna(row.get('PLAYER_ID')) else None,
                "position": row.get('POSITION_GROUP', 'Wing')
            }

        return {"similar": results, "target": target_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# API ENDPOINTS - DIAMOND FINDER
# =============================================================================

@app.get("/api/diamond-finder/players")
def get_diamond_finder_players(season: str = Query(default=CURRENT_SEASON)):
    """Get players for diamond finder dropdown (sorted by salary)."""
    model_data = get_diamond_finder_model(season)
    if model_data is None:
        return {"players": []}

    df = model_data['df']
    df_sorted = df.sort_values('current_year_salary', ascending=False)

    players = []
    for _, row in df_sorted.iterrows():
        players.append({
            "name": row['player_name'],
            "salary": float(row.get('current_year_salary', 0)),
            "lebron": float(row.get('LEBRON', 0))
        })

    return {"players": players}


@app.get("/api/diamond-finder/find")
def find_diamond_replacements(
    player: str = Query(..., description="Player name to find replacements for"),
    season: str = Query(default=CURRENT_SEASON)
):
    """Find cheaper replacement players with similar production."""
    model_data = get_diamond_finder_model(season)
    if model_data is None:
        raise HTTPException(status_code=500, detail="Diamond finder model not available")

    df = model_data['df']
    target_mask = df['player_name'] == player

    if not target_mask.any():
        raise HTTPException(status_code=404, detail=f"Player '{player}' not found")

    target_row = df[target_mask].iloc[0]

    replacements = data_processing.find_replacement_players(
        player, df, model_data['model'], model_data['scaler'],
        model_data['feature_info'], max_results=8
    )

    # Convert numpy types to native Python types
    replacements = convert_numpy_types(replacements)

    target_info = {
        "name": player,
        "salary": float(target_row.get('current_year_salary', 0)),
        "lebron": float(target_row.get('LEBRON', 0)),
        "archetype": target_row.get('Offensive Archetype', 'Unknown'),
        "defense_role": target_row.get('Defensive Role', 'Unknown'),
        "player_id": int(target_row.get('PLAYER_ID')) if pd.notna(target_row.get('PLAYER_ID')) else None
    }

    return {"target": target_info, "replacements": replacements}


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_caching_process():
    """Runs the full data caching pipeline."""
    print("Starting Sieve data caching process...")
    # Add an import statement for the CURRENT_SEASON
    from .config import CURRENT_SEASON
    # Define seasons to process
    seasons_to_process = data_processing.get_season_list(start_year=2022) # Process last few seasons
    seasons_to_process.append(CURRENT_SEASON)
    seasons_to_process = sorted(list(set(seasons_to_process)), reverse=True)

    for season in seasons_to_process:
        print(f"\\n{'='*60}")
        print(f"PROCESSING SEASON: {season}")
        print(f"{'='*60}\\n")

        # 1. Fetch all raw data from APIs
        data_processing.fetch_standings(force_refresh=True, season=season)
        data_processing.fetch_nba_advanced_stats(force_refresh=True, season=season)
        data_processing.fetch_player_pergame_stats(force_refresh=True, season=season)
        
        try:
            data_processing.fetch_lineup_stats(force_refresh=True, season=season)
        except Exception as e:
            print(f"Could not fetch lineup stats for {season}: {e}")

        # 2. Load and merge raw data from local files and DB
        lebron_file = f'data/LEBRON_{season}.csv' if season != '2024-25' else 'data/LEBRON.csv'
        df_players = data_processing.load_and_merge_data(lebron_file=lebron_file, season=season, from_db=False) # from_db=False to force re-merge

        if df_players.empty:
            print(f"No player data found for {season}. Skipping.")
            continue

        # 3. Calculate player value metrics and save
        df_players_processed = data_processing.calculate_player_value_metrics(df_players, season=season)

        # 4. Calculate team metrics and save
        df_teams = data_processing.calculate_team_metrics(df_players_processed, season=season)
        
        # 5. Add team logos and save final team data
        if not df_teams.empty:
            df_teams_with_logos = data_processing.add_team_logos(df_teams)
            cache.save_team_efficiency(df_teams_with_logos, season=season)
        
        print(f"Successfully processed and cached all data for {season}")

    # 6. Process historical data for similarity model
    print("\\nProcessing historical data for similarity model...")
    df_history = data_processing.fetch_historical_data(force_refresh=True)
    if not df_history.empty:
        model, scaler, df, features = data_processing.build_similarity_model(df_history)
        if model:
            cache.save_similarity_model({
                'model': model,
                'scaler': scaler,
                'df': df,
                'feature_info': features
            })
            print("Saved similarity model to cache.")

    print("\\nPipeline finished. All seasons processed and cached.")


def run_caching_process():
    """Runs the full data caching pipeline."""
    print("Starting Sieve data caching process...")
    # Add an import statement for the CURRENT_SEASON
    from .config import CURRENT_SEASON
    # Define seasons to process
    seasons_to_process = data_processing.get_season_list(start_year=2022) # Process last few seasons
    seasons_to_process.append(CURRENT_SEASON)
    seasons_to_process = sorted(list(set(seasons_to_process)), reverse=True)

    for season in seasons_to_process:
        print(f"\\n{'='*60}")
        print(f"PROCESSING SEASON: {season}")
        print(f"{'='*60}\\n")

        # 1. Fetch all raw data from APIs
        data_processing.fetch_standings(force_refresh=True, season=season)
        data_processing.fetch_nba_advanced_stats(force_refresh=True, season=season)
        data_processing.fetch_player_pergame_stats(force_refresh=True, season=season)
        
        try:
            data_processing.fetch_lineup_stats(force_refresh=True, season=season)
        except Exception as e:
            print(f"Could not fetch lineup stats for {season}: {e}")

        # 2. Load and merge raw data from local files and DB
        lebron_file = f'data/LEBRON_{season}.csv' if season != '2024-25' else 'data/LEBRON.csv'
        df_players = data_processing.load_and_merge_data(lebron_file=lebron_file, season=season, from_db=False) # from_db=False to force re-merge

        if df_players.empty:
            print(f"No player data found for {season}. Skipping.")
            continue

        # 3. Calculate player value metrics and save
        df_players_processed = data_processing.calculate_player_value_metrics(df_players, season=season)

        # 4. Calculate team metrics and save
        df_teams = data_processing.calculate_team_metrics(df_players_processed, season=season)
        
        # 5. Add team logos and save final team data
        if not df_teams.empty:
            df_teams_with_logos = data_processing.add_team_logos(df_teams)
            cache.save_team_efficiency(df_teams_with_logos, season=season)
        
        print(f"Successfully processed and cached all data for {season}")

    # 6. Process historical data for similarity model
    print("\\nProcessing historical data for similarity model...")
    df_history = data_processing.fetch_historical_data(force_refresh=True)
    if not df_history.empty:
        model, scaler, df, features = data_processing.build_similarity_model(df_history)
        if model:
            cache.save_similarity_model({
                'model': model,
                'scaler': scaler,
                'df': df,
                'feature_info': features
            })
            print("Saved similarity model to cache.")

    print("\\nPipeline finished. All seasons processed and cached.")


if __name__ == "__main__":
    import sys
    import uvicorn

    if "cache" in sys.argv:
        run_caching_process()
    else:
        print("Starting Sieve API server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)


