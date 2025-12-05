"""
Central configuration for Sieve NBA Analytics.
Contains season detection, constants, and shared configuration.
"""

from datetime import datetime


def get_current_season():
    """
    Returns the current NBA season string (e.g., '2024-25').
    
    Logic:
    - NBA season typically starts in October
    - If current month is Oct-Dec, season started this year
    - If current month is Jan-Sep, season started last year
    
    Examples:
    - November 2024 -> '2024-25'
    - February 2025 -> '2024-25'  
    - October 2025 -> '2025-26'
    """
    now = datetime.now()
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def get_previous_season():
    """Returns the previous NBA season string."""
    now = datetime.now()
    year = now.year if now.month >= 10 else now.year - 1
    prev_year = year - 1
    return f"{prev_year}-{str(prev_year + 1)[-2:]}"


def get_season_display():
    """Returns human-readable season (e.g., '2024-25 NBA Season')."""
    return f"{get_current_season()} NBA Season"


def get_season_years():
    """
    Returns tuple of (start_year, end_year) for current season.
    
    Example: For '2024-25' returns (2024, 2025)
    """
    season = get_current_season()
    start = int(season[:4])
    return (start, start + 1)


# Current season constant - use this throughout the app
CURRENT_SEASON = get_current_season()

# Historical data start year
HISTORICAL_START_YEAR = 2003

# Cache settings
DEFAULT_CACHE_HOURS = 24  # How long before cache is considered stale

# API settings
NBA_API_TIMEOUT = 60  # seconds
NBA_API_RETRY_COUNT = 3
NBA_API_RETRY_DELAY = 2  # seconds

# File paths
DATA_DIR = 'data'

# Primary data storage (Split SQLite databases)
PLAYERS_DB = f'{DATA_DIR}/sieve_players.db'  # Player stats, contracts, analysis
TEAMS_DB = f'{DATA_DIR}/sieve_teams.db'      # Team stats, standings, lineups

# Legacy unified database (deprecated)
CACHE_DB = f'{DATA_DIR}/sieve_cache.db'

# External raw input files (CSVs kept for manual updates)
LEBRON_FILE = f'{DATA_DIR}/LEBRON.csv'  # External: BBRef LEBRON metrics
BBREF_RAW_FILE = f'{DATA_DIR}/bbref_contracts_raw.csv'  # External: Raw BBRef contracts

# Legacy CSV path (deprecated - use DB via cache_manager)
CONTRACTS_FILE = f'{DATA_DIR}/basketball_reference_contracts.csv'

# Team abbreviation normalization
ABBR_NORMALIZATION = {
    'PHO': 'PHX',
    'CHO': 'CHA', 
    'BRK': 'BKN',
    'NOH': 'NOP',
    'TOT': 'UNK'  # "Total" for multi-team players
}


if __name__ == '__main__':
    # Quick test
    print(f"Current Season: {CURRENT_SEASON}")
    print(f"Previous Season: {get_previous_season()}")
    print(f"Display: {get_season_display()}")
    print(f"Season Years: {get_season_years()}")

