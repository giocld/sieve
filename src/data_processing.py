"""
Data processing utilities for Sieve NBA Analytics.
This module contains shared functions for loading raw data, merging datasets,
cleaning data, and calculating advanced metrics like Value Gap and Efficiency Index.
"""

import pandas as pd
import numpy as np
import os

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
