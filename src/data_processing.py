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
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats, leaguedashptstats

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

    # --- NEW: Add Player IDs for Headshots ---
    # We try to load the historical stats cache to get a mapping of Name -> ID
    historical_file = 'data/nba_historical_stats.csv'
    if os.path.exists(historical_file):
        try:
            df_hist = pd.read_csv(historical_file)
            if 'PLAYER_NAME' in df_hist.columns and 'PLAYER_ID' in df_hist.columns:
                # Create a mapping dictionary (Name -> ID)
                # We use the most recent ID found for a player
                id_map = df_hist.set_index('PLAYER_NAME')['PLAYER_ID'].to_dict()
                
                # Map IDs to the main dataframe
                df['PLAYER_ID'] = df['player_name'].map(id_map)
        except Exception as e:
            print(f"Warning: Could not load player IDs from history: {e}")
    
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


def fetch_historical_data(start_year=2003, cache_file='data/nba_historical_stats.csv', include_tracking=True):
    """
    Fetches historical player stats (Base + Advanced + Tracking) for multiple seasons.
    Calculates Relative True Shooting (rTS), era-adjusted metrics, and defensive profiles.
    
    Args:
        start_year (int): Starting year for historical data
        cache_file (str): Path to cache file
        include_tracking (bool): Whether to fetch tracking data (slower but more detailed)
    """
    if os.path.exists(cache_file):
        print(f"Loading historical data from {cache_file}...")
        return pd.read_csv(cache_file)



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
            
            # 4. Fetch Tracking Data (Optional - for ball-handling style)
            if include_tracking:
                try:
                    # Tracking data includes: TOUCHES, AVG_SEC_PER_TOUCH, AVG_DRIB_PER_TOUCH
                    tracking = leaguedashptstats.LeagueDashPtStats(
                        season=season,
                        per_mode_simple='PerGame',
                        player_or_team='Player'
                    )
                    df_tracking = tracking.get_data_frames()[0]
                    
                    # Select only tracking-specific columns to avoid duplicates
                    tracking_cols = ['PLAYER_ID', 'TOUCHES', 'AVG_SEC_PER_TOUCH', 
                                   'AVG_DRIB_PER_TOUCH', 'DIST_MILES', 'AVG_SPEED']
                    tracking_cols = [c for c in tracking_cols if c in df_tracking.columns]
                    
                    df_merged = pd.merge(
                        df_merged, 
                        df_tracking[tracking_cols], 
                        on='PLAYER_ID', 
                        how='left'
                    )
                    time.sleep(0.6)
                except Exception as e:
                    print(f"  Warning: Could not fetch tracking data for {season}: {e}")
            
            # 5. Calculate League Averages for Era Adjustment
            # We filter for qualified players to get a "rotation player" average
            qualified_players = df_merged[df_merged['GP'] > 10]
            
            if not qualified_players.empty:
                # True Shooting
                league_avg_ts = qualified_players['TS_PCT'].mean()
                
                # 3-Point Rate (era adjustment - accounts for 3-point revolution)
                league_avg_3pa = qualified_players['FG3A'].mean()
                
                # Pace proxy (for era normalization)
                league_avg_pts = qualified_players['PTS'].mean()
            else:
                league_avg_ts = 0.55
                league_avg_3pa = 2.0
                league_avg_pts = 15.0
            
            # 6. Calculate Relative/Era-Adjusted Metrics
            df_merged['LEAGUE_AVG_TS'] = league_avg_ts
            df_merged['LEAGUE_AVG_3PA'] = league_avg_3pa
            df_merged['LEAGUE_AVG_PTS'] = league_avg_pts
            
            # Scale percentages to 0-100 for readability
            df_merged['rTS'] = (df_merged['TS_PCT'] - league_avg_ts) * 100
            df_merged['USG_PCT'] = df_merged['USG_PCT'] * 100
            df_merged['AST_PCT'] = df_merged['AST_PCT'] * 100
            
            # Defensive rebounding percentage (already in Advanced stats)
            if 'DREB_PCT' in df_merged.columns:
                df_merged['DREB_PCT'] = df_merged['DREB_PCT'] * 100
            if 'OREB_PCT' in df_merged.columns:
                df_merged['OREB_PCT'] = df_merged['OREB_PCT'] * 100
            
            # 7. Calculate Additional Derived Metrics
            
            # 3-Point Attempt Rate (3PA / FGA) - shooting style
            df_merged['3PA_RATE'] = df_merged.apply(
                lambda x: x['FG3A'] / x['FGA'] if x['FGA'] > 0 else 0, axis=1
            )
            
            # 2-Point FG% (for inside/mid-range game)
            df_merged['FG2_PCT'] = df_merged.apply(
                lambda x: (x['FGM'] - x['FG3M']) / (x['FGA'] - x['FG3A']) 
                if (x['FGA'] - x['FG3A']) > 0 else 0, 
                axis=1
            )
            
            # Free Throw % (already in base stats as FT_PCT)
            # Just ensure it exists
            if 'FT_PCT' not in df_merged.columns:
                df_merged['FT_PCT'] = df_merged.apply(
                    lambda x: x['FTM'] / x['FTA'] if x['FTA'] > 0 else 0,
                    axis=1
                )
            
            # Turnover to Assist Ratio (decision-making quality)
            df_merged['TOV_AST_RATIO'] = df_merged.apply(
                lambda x: x['TOV'] / x['AST'] if x['AST'] > 0 else x['TOV'],
                axis=1
            )
            
            # Ball-handling style (if tracking data available)
            if 'TOUCHES' in df_merged.columns and 'AVG_DRIB_PER_TOUCH' in df_merged.columns:
                # High touches + high dribbles = ball-dominant creator
                # Low touches + low dribbles = catch-and-shoot / off-ball
                df_merged['BALL_DOMINANT'] = (
                    df_merged['TOUCHES'].fillna(0) * df_merged['AVG_DRIB_PER_TOUCH'].fillna(0)
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


def classify_position_group(row):
    """
    Classifies a player into a position group based on their statistical profile.
    
    Returns:
        str: 'guard', 'wing', or 'big'
    """
    # Thresholds for position classification
    # Guards: High AST%, Low REB%, High 3PA_RATE
    # Bigs: High REB%, Low AST%, Low 3PA_RATE  
    # Wings: Balanced
    
    ast_pct = row.get('AST_PCT', 0)
    reb = row.get('REB', 0)
    three_rate = row.get('3PA_RATE', 0)
    
    # Guard indicators: AST% > 20, REB < 5, or high 3PA rate with high AST
    if ast_pct > 20 or (ast_pct > 15 and reb < 5):
        return 'guard'
    
    # Big indicators: REB > 10, low 3PA rate, low AST%
    elif reb > 10 or (reb > 8 and three_rate < 0.15 and ast_pct < 15):
        return 'big'
    
    # Everything else is a wing
    else:
        return 'wing'


def build_similarity_model(df):
    """
    Trains an enhanced NearestNeighbors model with weighted features and position classification.
    
    Improvements:
    - 20+ features including defensive metrics, shooting profile, tracking data
    - Weighted feature importance (efficiency/style weighted higher than volume)
    - Cosine similarity for better style matching
    - Position group classification for filtering
    
    Returns:
        tuple: (model, scaler, df_filtered, feature_weights)
    """


    # Filter for qualified players (>= 15 games)
    df_filtered = df[df['GP'] >= 15].copy()
    
    # Handle missing values
    df_filtered = df_filtered.fillna(0)
    
    # ENHANCED FEATURE SET (20+ features)
    # Organized by category with weights
    
    feature_config = {
        # PRODUCTION (Volume stats - moderate weight)
        'PTS': 1.0,
        'REB': 1.0, 
        'AST': 1.0,
        'STL': 0.8,
        'BLK': 0.8,
        'TOV': 0.8,
        
        # EFFICIENCY/STYLE (High weight - defines HOW they play)
        'USG_PCT': 1.5,
        'rTS': 1.5,
        'AST_PCT': 1.5,
        '3PA_RATE': 1.5,
        'FT_PCT': 1.2,
        'FG2_PCT': 1.2,
        
        # DEFENSIVE PROFILE (Moderate-high weight)
        'DREB_PCT': 1.3,
        'OREB_PCT': 1.0,
        'DEF_RATING': 1.2,
        
        # PLAYMAKING QUALITY (Moderate weight)
        'TOV_AST_RATIO': 1.0,
        
        # TRACKING DATA (if available - moderate weight)
        'TOUCHES': 1.1,
        'AVG_SEC_PER_TOUCH': 1.1,
        'AVG_DRIB_PER_TOUCH': 1.1,
        'BALL_DOMINANT': 1.2,
    }
    
    # Filter to only available features
    available_features = []
    feature_weights = []
    
    for feature, weight in feature_config.items():
        if feature in df_filtered.columns:
            available_features.append(feature)
            feature_weights.append(weight)
    
    if not available_features:
        print("ERROR: No features available for similarity model")
        return None, None, None, None
    
    print(f"Building similarity model with {len(available_features)} features:")
    print(f"  Features: {', '.join(available_features)}")
    
    # Extract feature matrix
    X = df_filtered[available_features].values
    
    # Step 1: Normalize features with StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Apply feature weights
    # Multiply each feature column by its weight
    feature_weights_array = np.array(feature_weights).reshape(1, -1)
    X_weighted = X_scaled * feature_weights_array
    
    print(f"  Applied weighted features (efficiency/style prioritized)")
    
    # Step 3: Classify position groups
    df_filtered['POSITION_GROUP'] = df_filtered.apply(classify_position_group, axis=1)
    
    position_counts = df_filtered['POSITION_GROUP'].value_counts()
    print(f"  Position distribution: Guard={position_counts.get('guard', 0)}, "
          f"Wing={position_counts.get('wing', 0)}, Big={position_counts.get('big', 0)}")
    
    # Step 4: Train KNN with COSINE similarity
    # Cosine measures direction/pattern regardless of magnitude
    # Better for "type of player" vs "stats magnitude"
    model = NearestNeighbors(
        n_neighbors=30,  # Increased to allow for position filtering
        metric='cosine',  # CHANGED from euclidean
        n_jobs=1
    )
    model.fit(X_weighted)
    
    print(f"  Trained KNN model with cosine similarity")
    
    # Return model, scaler, filtered data, and feature info
    return model, scaler, df_filtered, {
        'features': available_features,
        'weights': feature_weights
    }


def find_similar_players(player_name, season, df_history, model, scaler, feature_info=None, exclude_self=True):
    """
    Finds the top 5 similar players with position-aware filtering and improved scoring.
    
    Args:
        player_name (str): Target player name
        season (str): Target season
        df_history (pd.DataFrame): Historical player data
        model: Trained KNN model
        scaler: Trained scaler
        feature_info (dict): Dictionary with 'features' and 'weights' keys
        exclude_self (bool): Whether to exclude other seasons of the same player
        
    Returns:
        list: List of similar player dictionaries
    """
    # Find the target player's row
    target_row = df_history[
        (df_history['PLAYER_NAME'] == player_name) & 
        (df_history['SEASON_ID'] == season)
    ]
    
    if target_row.empty:
        return []
    
    target_row = target_row.iloc[0]
    
    # Get feature list
    if feature_info and 'features' in feature_info:
        available_features = feature_info['features']
        feature_weights = np.array(feature_info['weights']).reshape(1, -1)
    else:
        # Fallback to basic features
        available_features = [
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'USG_PCT', 'rTS', 'AST_PCT', '3PA_RATE'
        ]
        available_features = [f for f in available_features if f in df_history.columns]
        feature_weights = np.ones((1, len(available_features)))
    
    # Extract and transform target stats
    target_stats = target_row[available_features].values.reshape(1, -1)
    target_scaled = scaler.transform(target_stats)
    target_weighted = target_scaled * feature_weights
    
    # Get target position group
    target_position = target_row.get('POSITION_GROUP', 'wing')
    
    # Find neighbors
    distances, indices = model.kneighbors(target_weighted)
    
    # Define compatible position groups
    # Guards can match guards and wings
    # Wings can match anyone (most versatile)
    # Bigs can match bigs and wings
    position_compatibility = {
        'guard': ['guard', 'wing'],
        'wing': ['guard', 'wing', 'big'],
        'big': ['big', 'wing']
    }
    compatible_positions = position_compatibility.get(target_position, ['guard', 'wing', 'big'])
    
    results = []
    
    # Iterate through neighbors
    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        
        match_row = df_history.iloc[idx]
        match_name = match_row['PLAYER_NAME']
        match_season = match_row['SEASON_ID']
        match_position = match_row.get('POSITION_GROUP', 'wing')
        
        # 1. ALWAYS exclude the exact same player-season (the query itself)
        if match_name == player_name and match_season == season:
            continue
            
        # 2. If "Exclude Self" is checked, exclude ALL seasons of this player
        if exclude_self and match_name == player_name:
            continue
        
        # 3. Position filtering - only include compatible positions
        if match_position not in compatible_positions:
            continue
            
        # Calculate Match Score (0-100 scale, but non-linear to spread out differences)
        # Problem: Cosine distances for similar players are VERY small (0.01 - 0.2)
        # Old formula: (1 - dist/2) * 100 gave 95-100% for everything
        # 
        # New approach: Use exponential decay to spread scores better
        # - Very close (dist < 0.05): 90-100 score
        # - Close (dist 0.05-0.15): 75-90 score  
        # - Moderate (dist 0.15-0.30): 60-75 score
        # - Far (dist > 0.30): < 60 score
        match_score = 100 * np.exp(-dist * 5)  # Exponential decay with factor 5
        match_score = max(0, min(100, match_score))  # Clamp to [0, 100]
        
        results.append({
            'Player': match_name,
            'Season': match_season,
            'id': match_row['PLAYER_ID'],
            'MatchScore': round(match_score, 1),
            'Distance': round(dist, 4),  # Include raw distance for debugging
            'Position': match_position.title(),
            'Stats': match_row[available_features].to_dict()
        })
        
        # Stop after 5 matches
        if len(results) >= 5:
            break
        
    return results