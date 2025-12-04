# Sieve Cache System

## Overview

The Sieve cache system consolidates all cached data into a single SQLite database (`data/sieve_cache.db`) instead of multiple CSV files. This provides:

- **Single file management** - One database file instead of 10+ CSVs
- **Queryable data** - Filter without loading entire datasets into memory
- **Metadata tracking** - Know when each cache was last updated
- **Type safety** - SQLite handles data types consistently
- **Atomic operations** - Safe concurrent read/write access

## Architecture

```
data/
  sieve_cache.db          # Unified SQLite database
  LEBRON.csv              # Source data (not cached)
  basketball_reference_contracts.csv  # Source data (not cached)
```

### Database Schema

The database contains these tables:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `lineups` | Duo/trio lineup performance | `_group_size`, `_team_filter`, `_season` |
| `player_stats` | Historical player statistics | `SEASON_ID`, `PLAYER_ID` |
| `team_stats` | Team advanced metrics | `_stat_type`, `_season` |
| `standings` | Team win/loss records | `_season` |
| `cache_metadata` | Tracks cache freshness | `cache_name`, `last_updated` |

## Usage

### Python API

```python
from src.cache_manager import cache

# Check cache status
cache.print_status()

# Load data (automatically uses cache)
df = cache.load_lineups(group_size=2, season='2024-25')
df = cache.load_player_stats()
df = cache.load_team_stats(stat_type='advanced')
df = cache.load_standings(season='2024-25')

# Save data to cache
cache.save_lineups(df, group_size=2, team_abbr=None, season='2024-25')
cache.save_player_stats(df, season='2024-25')
cache.save_team_stats(df, stat_type='advanced', season='2024-25')
cache.save_standings(df, season='2024-25')

# Check if cache is fresh (within 24 hours)
if cache.is_cache_fresh('lineups_2man_ALL', max_age_hours=24):
    print("Cache is up to date")

# Clear specific cache
cache.clear_cache('lineups_2man_ALL')

# Clear all caches
cache.clear_cache()
```

### CLI Commands

```bash
# Show cache status
python -m src.manage_cache status

# Migrate existing CSV files to SQLite
python -m src.manage_cache migrate

# List all cached datasets
python -m src.manage_cache list

# Clear all caches
python -m src.manage_cache clear

# Clear specific cache
python -m src.manage_cache clear --name lineups_2man_ALL
```

## Adding New Data Types

To add a new data type to the cache system:

### Step 1: Add save/load methods to `cache_manager.py`

```python
# In CacheManager class

def save_my_data(self, df, category='default', season='2024-25'):
    """Save custom data to cache."""
    if df.empty:
        return
    
    df = df.copy()
    df['_category'] = category
    df['_season'] = season
    
    cache_name = f'my_data_{category}'
    
    with self._get_connection() as conn:
        # Check if table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='my_data'"
        )
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            conn.execute(
                'DELETE FROM my_data WHERE _category = ? AND _season = ?',
                (category, season)
            )
        
        df.to_sql('my_data', conn, if_exists='append', index=False)
        self._update_metadata(conn, cache_name, season, len(df))
        conn.commit()
    
    print(f"Cached {len(df)} records to {cache_name}")

def load_my_data(self, category='default', season='2024-25'):
    """Load custom data from cache."""
    try:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='my_data'"
            )
            if not cursor.fetchone():
                return None
            
            df = pd.read_sql(
                'SELECT * FROM my_data WHERE _category = ? AND _season = ?',
                conn, params=[category, season]
            )
            
            if df.empty:
                return None
            
            # Drop internal columns
            df = df.drop(columns=['_category', '_season'], errors='ignore')
            return df
            
    except Exception as e:
        print(f"Error loading my_data: {e}")
        return None
```

### Step 2: Update the data fetching function

```python
# In data_processing.py

def fetch_my_data(category='default', force_refresh=False, season='2024-25'):
    """Fetch data with caching support."""
    
    # Check cache first
    if not force_refresh:
        df = cache.load_my_data(category, season)
        if df is not None and not df.empty:
            print(f"Loaded my_data from cache")
            return df
    
    # Fetch from API/source
    try:
        df = some_api_call()
        
        # Save to cache
        cache.save_my_data(df, category, season)
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
```

### Step 3: Add migration support (optional)

```python
# In CacheManager.migrate_from_csv()

# Add to the migrate_from_csv method:
my_data_file = os.path.join(data_dir, 'my_data_cache.csv')
if os.path.exists(my_data_file):
    try:
        df = pd.read_csv(my_data_file)
        if not df.empty:
            self.save_my_data(df, 'default', '2024-25')
            migrated.append("My custom data")
    except Exception as e:
        print(f"Error migrating my_data: {e}")
```

## Internal Column Conventions

All cached tables use underscore-prefixed columns for metadata:

| Column | Purpose |
|--------|---------|
| `_season` | NBA season string (e.g., '2024-25') |
| `_group_size` | Lineup size (2, 3, 5) |
| `_team_filter` | Team abbreviation or 'ALL' |
| `_stat_type` | Type of stats (advanced, base, etc.) |

These columns are automatically stripped when loading data.

## Best Practices

1. **Always check cache first** - Use `force_refresh=False` by default
2. **Use appropriate cache keys** - Include season, team, stat_type as needed
3. **Set reasonable cache ages** - 24 hours for standings, longer for historical data
4. **Clean up after migration** - Delete old CSV files once verified

## Migrating Remaining CSVs

Current source files that are NOT cached (and shouldn't be):
- `LEBRON.csv` - Source data, updated manually
- `basketball_reference_contracts.csv` - Source data, updated manually

These are source files, not caches. They should remain as CSVs since they're manually updated.

### Files that CAN be deleted after migration:

```bash
# After verifying the dashboard works:
rm data/nba_advanced_stats_cache.csv
rm data/nba_historical_stats.csv
rm data/nba_lineups_cache*.csv
rm data/nba_standings_cache.csv
```

## Troubleshooting

### Cache not loading
```python
# Force refresh to bypass cache
df = fetch_lineup_data(force_refresh=True)
```

### Clear corrupted cache
```bash
python -m src.manage_cache clear
```

### Check what's cached
```bash
python -m src.manage_cache status
```

### Database too large
The SQLite database may grow over time. To compact:
```python
from src.cache_manager import cache
with cache._get_connection() as conn:
    conn.execute('VACUUM')
```

