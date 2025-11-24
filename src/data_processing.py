"""
Data processing utilities for Sieve NBA Analytics.
This module contains shared functions for loading raw data, merging datasets,
cleaning data, and calculating advanced metrics like Value Gap and Efficiency Index.
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats

# Suppress harmless multiprocessing cleanup warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Mapping of full team names to their standard 3-letter abbreviations.
# This ensures consistency across different data sources (e.g., LEBRON data vs Contract data).
TEAM_ABBR_MAP = {
    'Cleveland Cavaliers': 'CLE', 'Oklahoma City Thunder': 'OKC', 'Boston Celtics': 'BOS',
    'Houston Rockets': 'HOU', 'Los Angeles Lakers': 'LAL', 'New York Knicks': 'NYK',
    'Indiana Pacers': 'IND', 'Denver Nuggets': 'DEN', 'LA Clippers': 'LAC',
    'Milwaukee Bucks': 'MIL', 'Detroit Pistons': 'DET', 'Minnesota Timberwolves': 'MIN',
    'Orlando Magic': 'ORL', 'Golden State Warriors': 'GSW', 'Memphis Grizzlies': 'MEM',
    'Atlanta Hawks': 'ATL', 'Chicago Bulls': 'CHI', 'Sacramento Kings': 'SAC',
    'Dallas Mavericks': 'DAL', 'Miami Heat': 'MIA', 'Phoenix Suns': 'PHX',
    'Toronto Raptors': 'TOR', 'Portland Trail Blazers': 'POR', 'Brooklyn Nets': 'BKN',
    'Philadelphia 76ers': 'PHI', 'San Antonio Spurs': 'SAS', 'Charlotte Hornets': 'CHA',
    'New Orleans Pelicans': 'NOP', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

# Normalization map to handle non-standard abbreviations found in some datasets.
# For example, Basketball Reference might use 'CHO' for Charlotte instead of 'CHA'.
ABBR_NORMALIZATION = {
    'PHO': 'PHX', 'CHO': 'CHA', 'BRK': 'BKN', 'NOH': 'NOP', 'TOT': 'UNK'
}


def load_and_merge_data(lebron_file='data/LEBRON.csv', contracts_file='data/basketball_reference_contracts.csv'):
    """
    Loads player impact data (LEBRON) and contract data, merges them into a single dataset,
    and performs initial data cleaning and type conversion.

    Args:
        lebron_file (str): Path to the CSV file containing LEBRON impact metrics.
        contracts_file (str): Path to the CSV file containing player contract details.

    Returns:
        pd.DataFrame: A merged DataFrame containing both performance and financial data for each player.
    """
    # Verify that source files exist before attempting to load
    if not os.path.exists(lebron_file) or not os.path.exists(contracts_file):
        raise FileNotFoundError(f"Missing {lebron_file} or {contracts_file}")
    
    df_lebron = pd.read_csv(lebron_file)
    df_contracts = pd.read_csv(contracts_file)
    
    # Standardize the player name column to 'player_name' to ensure a successful merge
    if 'Player' in df_lebron.columns:
        df_lebron = df_lebron.rename(columns={'Player': 'player_name'})
    
    # Convert key numerical columns to numeric types, coercing errors to NaN
    # This prevents string formatting issues from breaking calculations later
    for col in ['LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON']:
        if col in df_lebron.columns:
            df_lebron[col] = pd.to_numeric(df_lebron[col], errors='coerce')
    
    # Create a combined 'archetype' string (e.g., "Shot Creator / Wing Defender")
    # This provides a more granular description of the player's role than position alone
    if 'Offensive Archetype' in df_lebron.columns and 'Defensive Role' in df_lebron.columns:
        df_lebron['archetype'] = (
            df_lebron['Offensive Archetype'].fillna('Unknown').astype(str) + 
            ' / ' + 
            df_lebron['Defensive Role'].fillna('Unknown').astype(str)
        )
    else:
        df_lebron['archetype'] = 'Unknown'
    
    # Merge the datasets using an inner join on the player's name.
    # Players not found in both datasets will be excluded.
    df = pd.merge(df_lebron, df_contracts, on='player_name', how='inner')
    
    # Remove duplicate entries for the same player, keeping the first occurrence.
    # This handles cases where a player might be listed multiple times (e.g., traded players).
    df = df.drop_duplicates(subset=['player_name'], keep='first')
    
    # Handle missing salary data for the current year.
    # If 'current_year_salary' is missing, we fallback to 'year_4' (Guaranteed Amount).
    # This is crucial because some contract datasets might not explicitly list the current year column correctly.
    if 'current_year_salary' in df.columns:
        nan_mask = df['current_year_salary'].isna()
        if 'year_4' in df.columns:
            df.loc[nan_mask, 'current_year_salary'] = df.loc[nan_mask, 'year_4']
        
        # If still missing, fill with 0 to allow for calculations without errors
        df['current_year_salary'] = df['current_year_salary'].fillna(0)
    
    return df



def calculate_player_value_metrics(df):
    """
    Calculates the 'Value Gap' for each player.
    
    The Value Gap is a proprietary metric that quantifies the difference between a player's
    on-court impact and their financial cost.
    
    Methodology:
    1. Normalize 'current_year_salary' to a 0-100 scale.
    2. Normalize 'LEBRON' impact to a 0-100 scale.
    3. Value Gap = Normalized Impact - Normalized Salary.
    
    Interpretation:
    - Positive Gap: Player is Underpaid (providing more value than they cost).
    - Negative Gap: Player is Overpaid (costing more than the value they provide).

    Args:
        df (pd.DataFrame): DataFrame containing player data with 'current_year_salary' and 'LEBRON'.

    Returns:
        pd.DataFrame: The input DataFrame with added columns: 'salary_norm', 'impact_norm', 'value_gap'.
    """
    df = df.copy()
    
    # Ensure required columns exist before calculation
    if 'current_year_salary' in df.columns and 'LEBRON' in df.columns:
        valid_salary = df['current_year_salary'].dropna()
        valid_lebron = df['LEBRON'].dropna()
        
        # Only proceed if we have valid data points
        if len(valid_salary) > 0 and len(valid_lebron) > 0:
            # Normalize Salary (0 to 100)
            salary_min = valid_salary.min()
            salary_max = valid_salary.max()
            df['salary_norm'] = 100 * (df['current_year_salary'] - salary_min) / (salary_max - salary_min)
            
            # Normalize Impact (0 to 100)
            lebron_min = valid_lebron.min()
            lebron_max = valid_lebron.max()
            df['impact_norm'] = 100 * (df['LEBRON'] - lebron_min) / (lebron_max - lebron_min)
            
            # Calculate the Gap
            df['value_gap'] = df['impact_norm']*1.4 - df['salary_norm']*0.9 - 10
        else:
            df['value_gap'] = 0
    else:
        df['value_gap'] = 0
    
    return df


def calculate_team_metrics(df_players, standings_file='data/nba_standings_cache.csv'):
    """
    Aggregates individual player data to the team level and calculates team efficiency metrics.
    
    This function performs the following steps:
    1. Loads team standings (Wins/Losses).
    2. Maps player teams to standard abbreviations.
    3. Aggregates player salaries and WAR to get Team Totals.
    4. Calculates the 'Efficiency Index' for each team.

    Args:
        df_players (pd.DataFrame): DataFrame containing individual player data.
        standings_file (str): Path to the CSV file containing cached NBA standings.

    Returns:
        pd.DataFrame: A DataFrame where each row is a Team, containing aggregated metrics.
    """
    # 1. Load Standings Data
    if not os.path.exists(standings_file):
        print(f"Warning: {standings_file} not found. Team metrics will be incomplete.")
        df_standings = pd.DataFrame()
    else:
        df_standings = pd.read_csv(standings_file)
        df_standings.columns = df_standings.columns.str.strip()
        
        # Construct Full Name if missing, to facilitate mapping to abbreviations
        if 'FullName' not in df_standings.columns and 'TeamCity' in df_standings.columns and 'TeamName' in df_standings.columns:
            df_standings['FullName'] = df_standings['TeamCity'] + ' ' + df_standings['TeamName']
        
        # Map full names to 3-letter abbreviations
        df_standings['Abbrev'] = df_standings['FullName'].map(TEAM_ABBR_MAP)
        df_standings = df_standings.dropna(subset=['Abbrev'])
    
    # 2. Normalize Team Abbreviations in Player Data
    # Players might have team codes like 'TOT' (Total) or non-standard codes.
    # We need to normalize AFTER splitting to handle multi-team entries like 'BRK/LAL'
    if 'Team(s)' in df_players.columns:
        df_players = df_players.copy()
        # Handle multi-team players (e.g., 'CLE/LAL') by taking the first team listed
        # Then normalize the abbreviation (e.g., 'BRK' -> 'BKN')
        def normalize_team(team_str):
            if not isinstance(team_str, str):
                return 'UNK'
            # Split on '/' and take the first team
            first_team = str(team_str).split('/')[0].strip()
            # Apply normalization to the first team code
            normalized = ABBR_NORMALIZATION.get(first_team, first_team)
            return normalized
        
        df_players['Team_Main'] = df_players['Team(s)'].apply(normalize_team)
    else:
        df_players = df_players.copy()
        df_players['Team_Main'] = 'UNK'
    
    # 3. Aggregate Data by Team
    df_teams = df_players.groupby('Team_Main').agg({
        'current_year_salary': 'sum',  # Total Payroll
        'LEBRON WAR': 'sum',           # Total Wins Above Replacement
        'LEBRON': 'mean',              # Average Player Impact
        'player_name': 'count'         # Roster Size (in dataset)
    }).reset_index()
    
    # Rename columns for clarity
    df_teams = df_teams.rename(columns={
        'current_year_salary': 'Total_Payroll',
        'LEBRON WAR': 'Total_WAR',
        'Team_Main': 'Abbrev'
    })
    
    # 4. Merge with Standings (Wins/Losses)
    if not df_standings.empty:
        df_teams = pd.merge(df_teams, df_standings[['Abbrev', 'WINS', 'LOSSES']], on='Abbrev', how='left')
    
    # Filter out invalid teams or teams with missing data
    df_teams = df_teams[
        (df_teams['Total_Payroll'] > 0) & 
        (df_teams['Abbrev'] != 'UNK') & 
        (df_teams['WINS'].notna() if 'WINS' in df_teams.columns else True)
    ].copy()
    
    # 5. Calculate Efficiency Metrics
    if not df_teams.empty and 'WINS' in df_teams.columns:
        # Cost Per Win: How much payroll was spent for each win
        df_teams['Cost_Per_Win'] = df_teams.apply(
            lambda row: row['Total_Payroll'] / row['WINS'] if row['WINS'] > 0 else 0, 
            axis=1
        )
        
        # Efficiency Index Calculation:
        # We use Z-scores to compare teams relative to the league average.
        # We weight Wins (2.0) higher than Payroll (1.0) to prioritize on-court success.
        # Formula: Efficiency = (2.0 * Z_Wins) - Z_Payroll
        
        wins_mean = df_teams['WINS'].mean()
        wins_std = df_teams['WINS'].std()
        payroll_mean = df_teams['Total_Payroll'].mean()
        payroll_std = df_teams['Total_Payroll'].std()
        
        if wins_std > 0 and payroll_std > 0:
            wins_z = (df_teams['WINS'] - wins_mean) / wins_std
            payroll_z = (df_teams['Total_Payroll'] - payroll_mean) / payroll_std
            df_teams['Efficiency_Index'] = (2.0 * wins_z) - payroll_z
        else:
            df_teams['Efficiency_Index'] = 0
    
    # Add formatted strings for display purposes (e.g., "$150.5M")
    df_teams['Payroll_Display'] = df_teams['Total_Payroll'].apply(lambda x: f"${x/1_000_000:.1f}M")
    if 'Cost_Per_Win' in df_teams.columns:
        df_teams['CPW_Display'] = df_teams['Cost_Per_Win'].apply(
            lambda x: f"${x/1_000_000:.2f}M" if x > 0 else "N/A"
        )
    
    return df_teams

def get_team_logo_url(team_id):
    """
    Generates the official NBA CDN URL for a team's logo based on their Team ID.
    
    Args:
        team_id (int): The NBA Team ID.
        
    Returns:
        str: The URL to the SVG logo.
    """
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"

def add_team_logos(df_teams):
    """
    Fetches NBA Team IDs and adds a 'Logo_URL' column to the teams DataFrame.
    
    This function uses the 'nba_api' library to get the official Team IDs,
    maps them to our team abbreviations, and then generates the logo URLs.
    
    Args:
        df_teams (pd.DataFrame): DataFrame containing team data with an 'Abbrev' column.
        
    Returns:
        pd.DataFrame: The input DataFrame with an added 'Logo_URL' column.
    """
    from nba_api.stats.static import teams
    nba_teams = teams.get_teams()
    
    # Create a mapping dictionary from Abbreviation to Team ID
    abbr_to_id = {}
    for t in nba_teams:
        abbr_to_id[t['abbreviation']] = t['id']
        
    # Manually handle special cases or historical abbreviations
    abbr_to_id['PHX'] = abbr_to_id.get('PHX', 1610612756) # Phoenix Suns
    abbr_to_id['BKN'] = abbr_to_id.get('BKN', 1610612751) # Brooklyn Nets
    abbr_to_id['CHA'] = abbr_to_id.get('CHA', 1610612766) # Charlotte Hornets
    abbr_to_id['NOP'] = abbr_to_id.get('NOP', 1610612740) # New Orleans Pelicans
    abbr_to_id['UTA'] = abbr_to_id.get('UTA', 1610612762) # Utah Jazz
    
    # Map IDs to the DataFrame and generate URLs
    df_teams['TeamID'] = df_teams['Abbrev'].map(abbr_to_id)
    df_teams['Logo_URL'] = df_teams['TeamID'].apply(lambda x: get_team_logo_url(int(x)) if pd.notna(x) else None)
    
    return df_teams


def fetch_nba_advanced_stats(cache_file='data/nba_advanced_stats_cache.csv'):
    """
    Fetches Advanced Team Stats from NBA API and caches them.
    
    Returns:
        pd.DataFrame: DataFrame with advanced stats.
    """
    if os.path.exists(cache_file):
        # Check if cache is recent (optional, skipping for now)
        return pd.read_csv(cache_file)
        
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced')
        df = stats.get_data_frames()[0]
        df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        print(f"Error fetching NBA API data: {e}")
        return pd.DataFrame()


def get_team_radar_data(team_abbr):
    """
    Prepares data for the 'Missing Piece' Radar Chart.
    
    Calculates percentiles for key metrics:
    - Offense (OFF_RATING)
    - Defense (DEF_RATING) - Inverted
    - Rebounding (REB_PCT)
    - Playmaking (AST_PCT)
    - Shooting (TS_PCT)
    
    Args:
        team_abbr (str): Team abbreviation (e.g., 'BOS').
        
    Returns:
        dict: Dictionary with metrics and percentiles for the team.
    """
    df = fetch_nba_advanced_stats()
    if df.empty:
        return None
        
    # Map abbreviation to Team Name if necessary, or use ID
    # The API returns TEAM_NAME. We need to match our 'team_abbr'.
    # We can use our TEAM_ABBR_MAP inverted or just fuzzy match.
    
    # Invert the map to get Full Name from Abbr
    abbr_to_name = {v: k for k, v in TEAM_ABBR_MAP.items()}
    
    # Handle edge cases
    abbr_to_name['PHX'] = 'Phoenix Suns'
    abbr_to_name['BKN'] = 'Brooklyn Nets'
    abbr_to_name['CHA'] = 'Charlotte Hornets'
    abbr_to_name['NOP'] = 'New Orleans Pelicans'
    abbr_to_name['UTA'] = 'Utah Jazz'
    
    full_name = abbr_to_name.get(team_abbr)
    
    if not full_name:
        return None
        
    # Find the team row
    team_row = df[df['TEAM_NAME'] == full_name]
    if team_row.empty:
        return None
        
    # Calculate Percentiles for the whole league
    metrics = {
        'Offense': 'OFF_RATING',
        'Defense': 'DEF_RATING',
        'Rebounding': 'REB_PCT',
        'Playmaking': 'AST_PCT',
        'Shooting': 'TS_PCT'
    }
    
    radar_data = {}
    
    for label, col in metrics.items():
        # Calculate percentile rank (0-100)
        # For Defense, Lower is Better, so we invert the rank
        if label == 'Defense':
            rank = df[col].rank(ascending=False, pct=True) * 100
        else:
            rank = df[col].rank(ascending=True, pct=True) * 100
            
        # Get the specific team's percentile
        team_val = rank[df['TEAM_NAME'] == full_name].values[0]
        radar_data[label] = team_val
        
    return radar_data

def get_season_list(start_year=2014):
    """
    Generates a list of NBA season strings (e.g., '2014-15') 
    from start_year up to the current active season.
    """
    current_date = datetime.now()
    
    # Logic: NBA season typically starts in October.
    # If it's Oct-Dec (Month >= 10), the season started this year (e.g., Oct 2025 is the 2025-26 season).
    # If it's Jan-Sept (Month < 10), the season started last year (e.g., Feb 2026 is the 2025-26 season).
    if current_date.month >= 10:
        current_season_start = current_date.year
    else:
        current_season_start = current_date.year - 1
        
    seasons = []
    for year in range(start_year, current_season_start + 1):
        # Format: "YYYY-YY" (e.g., 2024 -> "2024-25")
        next_year_short = str(year + 1)[-2:]
        season_str = f"{year}-{next_year_short}"
        seasons.append(season_str)
        
    return seasons


def fetch_historical_data(start_year=2003, cache_file='data/nba_historical_stats.csv'):
    """
    Fetches historical player stats (Base + Advanced) for multiple seasons.
    Calculates Relative True Shooting (rTS).
    """
    if os.path.exists(cache_file):
        print(f"Loading historical data from {cache_file}...")
        return pd.read_csv(cache_file)

    from nba_api.stats.endpoints import leaguedashplayerstats
    import time

    seasons = get_season_list(start_year=start_year)
    all_dfs = []
    
    print(f"Fetching historical data for {len(seasons)} seasons...")
    
    for season in seasons:
        try:
            print(f"Fetching {season}...")
            
            # 1. Fetch Base Stats (Per 100 Possessions) -> For Volume (PTS, REB, AST)
            stats_base = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='Per100Possessions',
                measure_type_detailed_defense='Base'
            )
            df_base = stats_base.get_data_frames()[0]
            time.sleep(0.6) # Respect API rate limits
            
            # 2. Fetch Advanced Stats -> For Efficiency/Style (USG%, TS%, AST%)
            stats_adv = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                measure_type_detailed_defense='Advanced'
            )
            df_adv = stats_adv.get_data_frames()[0]
            time.sleep(0.6)
            
            # 3. Merge them on PLAYER_ID
            # Suffixes handle columns that exist in both (like GP, MIN)
            df_merged = pd.merge(df_base, df_adv, on='PLAYER_ID', suffixes=('', '_ADV'))
            
            # 4. Calculate League Average TS% for this season
            # We filter for relevant players to get a "rotation player" average, not skewed by garbage time
            qualified_players = df_merged[df_merged['GP'] > 10]
            if not qualified_players.empty:
                league_avg_ts = qualified_players['TS_PCT'].mean()
            else:
                league_avg_ts = 0.55 # Fallback
            
            # 5. Calculate Relative True Shooting (rTS) and other derived metrics
            df_merged['LEAGUE_AVG_TS'] = league_avg_ts
            # Scale percentages to 0-100 for readability
            df_merged['rTS'] = (df_merged['TS_PCT'] - league_avg_ts) * 100
            df_merged['USG_PCT'] = df_merged['USG_PCT'] * 100
            df_merged['AST_PCT'] = df_merged['AST_PCT'] * 100
            
            # Calculate 3-Point Attempt Rate (3PA / FGA)
            # Handle division by zero
            df_merged['3PA_RATE'] = df_merged.apply(
                lambda x: x['FG3A'] / x['FGA'] if x['FGA'] > 0 else 0, axis=1
            )
            
            df_merged['SEASON_ID'] = season
            all_dfs.append(df_merged)
            
        except Exception as e:
            print(f"Error fetching {season}: {e}")
            continue
            
    if not all_dfs:
        return pd.DataFrame()
        
    full_history = pd.concat(all_dfs, ignore_index=True)
    
    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    full_history.to_csv(cache_file, index=False)
    
    return full_history


def build_similarity_model(df):
    """
    Trains a NearestNeighbors model on the provided dataframe.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    # e.g., > 15 games and > 10 mins/game 
    df_filtered = df[df['GP'] >= 15].copy()
    
    # Handle missing values
    df_filtered = df_filtered.fillna(0)
    
    # Features for similarity
    # We want a mix of Production (PTS, REB, AST) and Style (USG, rTS, 3PAr)
    # Note: Using Per100 stats for production
    features = [
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',  # Production (Per 100)
        'USG_PCT', 'rTS', 'AST_PCT', '3PA_RATE'    # Style / Efficiency
    ]
    
    # Ensure all features exist
    available_features = [f for f in features if f in df_filtered.columns]
    
    if not available_features:
        return None, None, None
        
    X = df_filtered[available_features].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KNN
    # n_neighbors=6 because the closest match is always the player themselves (distance=0)
    # n_jobs=1 to prevent resource exhaustion/crashes in limited environments
    model = NearestNeighbors(n_neighbors=20, metric='euclidean', n_jobs=1) # Increased neighbors to allow for filtering
    model.fit(X_scaled)
    
    return model, scaler, df_filtered


def find_similar_players(player_name, season, df_history, model, scaler, exclude_self=True):
    """
    Finds the top 5 similar players for a given player and season.
    """
    # Find the target player's row
    target_row = df_history[
        (df_history['PLAYER_NAME'] == player_name) & 
        (df_history['SEASON_ID'] == season)
    ]
    
    if target_row.empty:
        return []
        
    # Extract features
    features = [
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'USG_PCT', 'rTS', 'AST_PCT', '3PA_RATE'
    ]
    
    # Ensure features match training
    available_features = [f for f in features if f in df_history.columns]
    target_stats = target_row[available_features].values
    
    # Scale
    target_scaled = scaler.transform(target_stats)
    
    # Find neighbors
    distances, indices = model.kneighbors(target_scaled)
    
    results = []
    
    # Iterate through neighbors
    # indices[0] is the list of neighbor indices
    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        
        match_row = df_history.iloc[idx]
        match_name = match_row['PLAYER_NAME']
        match_season = match_row['SEASON_ID']
        
        # 1. ALWAYS exclude the exact same player-season (the query itself)
        if match_name == player_name and match_season == season:
            continue
            
        # 2. If "Exclude Self" is checked, exclude ALL seasons of this player
        if exclude_self and match_name == player_name:
            continue
            
        # Calculate a similarity score (0-100%)
        # Euclidean distance of 0 = 100% similar.
        # We need a heuristic to convert distance to %. 
        # A distance of ~5-6 is usually very different.
        similarity = max(0, 100 - (dist * 15)) # Heuristic conversion
        
        results.append({
            'Player': match_name,
            'Season': match_season,
            'id': match_row['PLAYER_ID'],
            'Similarity': round(similarity, 1),
            'Stats': match_row[available_features].to_dict()
        })
        
        # Stop after 5 matches
        if len(results) >= 5:
            break
        
    return results