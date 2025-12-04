"""
Cache Manager for Sieve NBA Analytics.
Provides a unified SQLite-based caching system for all data sources.
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
import json


class CacheManager:
    """
    Manages all cached data in a single SQLite database.
    
    Tables:
        - lineups: Duo/trio lineup data (with group_size column)
        - player_stats: Historical player statistics
        - team_stats: Team advanced stats
        - standings: Team standings/wins
        - cache_metadata: Tracks when each cache was last updated
    """
    
    def __init__(self, db_path='data/sieve_cache.db'):
        """Initialize the cache manager with database path."""
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Metadata table to track cache freshness
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    cache_name TEXT PRIMARY KEY,
                    last_updated TEXT,
                    season TEXT,
                    record_count INTEGER,
                    extra_info TEXT
                )
            ''')
            conn.commit()
    
    def _get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    # =========================================================================
    # METADATA OPERATIONS
    # =========================================================================
    
    def get_cache_info(self, cache_name):
        """
        Get metadata about a specific cache.
        
        Returns:
            dict or None: Cache metadata if exists
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM cache_metadata WHERE cache_name = ?',
                (cache_name,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    'cache_name': row[0],
                    'last_updated': datetime.fromisoformat(row[1]) if row[1] else None,
                    'season': row[2],
                    'record_count': row[3],
                    'extra_info': json.loads(row[4]) if row[4] else {}
                }
        return None
    
    def _update_metadata(self, conn, cache_name, season=None, record_count=0, extra_info=None):
        """Update cache metadata after writing data."""
        conn.execute('''
            INSERT OR REPLACE INTO cache_metadata 
            (cache_name, last_updated, season, record_count, extra_info)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            cache_name,
            datetime.now().isoformat(),
            season,
            record_count,
            json.dumps(extra_info) if extra_info else None
        ))
    
    def is_cache_fresh(self, cache_name, max_age_hours=24):
        """
        Check if a cache is fresh enough to use.
        
        Args:
            cache_name: Name of the cache
            max_age_hours: Maximum age in hours before cache is stale
            
        Returns:
            bool: True if cache exists and is fresh
        """
        info = self.get_cache_info(cache_name)
        if not info or not info['last_updated']:
            return False
        
        age = datetime.now() - info['last_updated']
        return age < timedelta(hours=max_age_hours)
    
    def list_caches(self):
        """List all cached data with metadata."""
        with self._get_connection() as conn:
            df = pd.read_sql('SELECT * FROM cache_metadata', conn)
        return df
    
    # =========================================================================
    # LINEUP DATA
    # =========================================================================
    
    def save_lineups(self, df, group_size, team_abbr=None, season='2024-25'):
        """
        Save lineup data to cache.
        
        Args:
            df: DataFrame with lineup data
            group_size: 2 for duos, 3 for trios
            team_abbr: Team filter (None for league-wide)
            season: NBA season string
        """
        if df.empty:
            return
        
        # Add metadata columns
        df = df.copy()
        df['_group_size'] = group_size
        df['_team_filter'] = team_abbr or 'ALL'
        df['_season'] = season
        
        cache_name = f'lineups_{group_size}man_{team_abbr or "ALL"}'
        
        with self._get_connection() as conn:
            # Check if table exists first
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='lineups'"
            )
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Delete existing data for this specific cache
                conn.execute('''
                    DELETE FROM lineups 
                    WHERE _group_size = ? AND _team_filter = ? AND _season = ?
                ''', (group_size, team_abbr or 'ALL', season))
            
            # Create table or append to existing
            df.to_sql('lineups', conn, if_exists='append', index=False)
            
            # Update metadata
            self._update_metadata(conn, cache_name, season, len(df), {
                'group_size': group_size,
                'team_abbr': team_abbr
            })
            conn.commit()
        
        print(f"Cached {len(df)} lineups to {cache_name}")
    
    def load_lineups(self, group_size, team_abbr=None, season='2024-25', min_minutes=0):
        """
        Load lineup data from cache.
        
        Args:
            group_size: 2 for duos, 3 for trios
            team_abbr: Team filter (None for league-wide)
            season: NBA season string
            min_minutes: Minimum total minutes filter
            
        Returns:
            DataFrame or None if not cached
        """
        cache_name = f'lineups_{group_size}man_{team_abbr or "ALL"}'
        
        try:
            with self._get_connection() as conn:
                # Check if table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='lineups'"
                )
                if not cursor.fetchone():
                    return None
                
                # Build query
                query = '''
                    SELECT * FROM lineups 
                    WHERE _group_size = ? AND _team_filter = ? AND _season = ?
                '''
                params = [group_size, team_abbr or 'ALL', season]
                
                df = pd.read_sql(query, conn, params=params)
                
                if df.empty:
                    return None
                
                # Calculate total minutes if needed
                if 'SUM_TIME_PLAYED' in df.columns:
                    df['TOTAL_MIN'] = df['SUM_TIME_PLAYED'] / 3000
                    if min_minutes > 0:
                        df = df[df['TOTAL_MIN'] >= min_minutes]
                
                # Drop internal columns before returning
                internal_cols = ['_group_size', '_team_filter', '_season']
                df = df.drop(columns=[c for c in internal_cols if c in df.columns], errors='ignore')
                
                print(f"Loaded {len(df)} lineups from cache: {cache_name}")
                return df
                
        except Exception as e:
            print(f"Cache read error: {e}")
            return None
    
    # =========================================================================
    # PLAYER HISTORICAL STATS
    # =========================================================================
    
    def save_player_stats(self, df, season=None):
        """Save historical player stats to cache."""
        if df.empty:
            return
        
        cache_name = 'player_historical_stats'
        
        with self._get_connection() as conn:
            if season:
                # Single season update - delete that season first, then append
                conn.execute(
                    'DELETE FROM player_stats WHERE SEASON_ID = ?',
                    (season,)
                )
                df.to_sql('player_stats', conn, if_exists='append', index=False)
            else:
                # Full bulk save - replace entire table
                df.to_sql('player_stats', conn, if_exists='replace', index=False)
            
            # Get total count
            count = conn.execute('SELECT COUNT(*) FROM player_stats').fetchone()[0]
            self._update_metadata(conn, cache_name, season, count)
            conn.commit()
        
        print(f"Cached {len(df)} player stat records")
    
    def load_player_stats(self, seasons=None, min_games=0):
        """
        Load historical player stats from cache.
        
        Args:
            seasons: List of seasons to load, or None for all
            min_games: Minimum games played filter
            
        Returns:
            DataFrame or None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='player_stats'"
                )
                if not cursor.fetchone():
                    return None
                
                if seasons:
                    placeholders = ','.join(['?' for _ in seasons])
                    query = f'SELECT * FROM player_stats WHERE SEASON_ID IN ({placeholders})'
                    df = pd.read_sql(query, conn, params=seasons)
                else:
                    df = pd.read_sql('SELECT * FROM player_stats', conn)
                
                if min_games > 0 and 'GP' in df.columns:
                    df = df[df['GP'] >= min_games]
                
                return df if not df.empty else None
                
        except Exception as e:
            print(f"Error loading player stats: {e}")
            return None
    
    # =========================================================================
    # TEAM STATS
    # =========================================================================
    
    def save_team_stats(self, df, stat_type='advanced', season='2024-25'):
        """Save team stats to cache."""
        if df.empty:
            return
        
        df = df.copy()
        df['_stat_type'] = stat_type
        df['_season'] = season
        
        cache_name = f'team_stats_{stat_type}'
        
        with self._get_connection() as conn:
            # Check if table exists first
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='team_stats'"
            )
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                conn.execute(
                    'DELETE FROM team_stats WHERE _stat_type = ? AND _season = ?',
                    (stat_type, season)
                )
            
            df.to_sql('team_stats', conn, if_exists='append', index=False)
            self._update_metadata(conn, cache_name, season, len(df))
            conn.commit()
        
        print(f"Cached {len(df)} team {stat_type} records")
    
    def load_team_stats(self, stat_type='advanced', season='2024-25'):
        """Load team stats from cache."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='team_stats'"
                )
                if not cursor.fetchone():
                    return None
                
                query = 'SELECT * FROM team_stats WHERE _stat_type = ? AND _season = ?'
                df = pd.read_sql(query, conn, params=[stat_type, season])
                
                if df.empty:
                    return None
                
                df = df.drop(columns=['_stat_type', '_season'], errors='ignore')
                return df
                
        except Exception as e:
            print(f"Error loading team stats: {e}")
            return None
    
    # =========================================================================
    # STANDINGS
    # =========================================================================
    
    def save_standings(self, df, season='2024-25'):
        """Save standings to cache."""
        if df.empty:
            return
        
        df = df.copy()
        df['_season'] = season
        
        with self._get_connection() as conn:
            # Check if table exists first
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='standings'"
            )
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                conn.execute('DELETE FROM standings WHERE _season = ?', (season,))
            
            df.to_sql('standings', conn, if_exists='append', index=False)
            self._update_metadata(conn, 'standings', season, len(df))
            conn.commit()
        
        print(f"Cached {len(df)} team standings")
    
    def load_standings(self, season='2024-25'):
        """Load standings from cache."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='standings'"
                )
                if not cursor.fetchone():
                    return None
                
                df = pd.read_sql(
                    'SELECT * FROM standings WHERE _season = ?',
                    conn, params=[season]
                )
                
                if df.empty:
                    return None
                
                df = df.drop(columns=['_season'], errors='ignore')
                return df
                
        except Exception as e:
            print(f"Error loading standings: {e}")
            return None
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def clear_cache(self, cache_name=None):
        """
        Clear cached data.
        
        Args:
            cache_name: Specific cache to clear, or None to clear all
        """
        with self._get_connection() as conn:
            if cache_name:
                # Parse cache name to determine table and filters
                if cache_name.startswith('lineups_'):
                    parts = cache_name.replace('lineups_', '').split('_')
                    group_size = int(parts[0].replace('man', ''))
                    team_filter = parts[1] if len(parts) > 1 else 'ALL'
                    conn.execute(
                        'DELETE FROM lineups WHERE _group_size = ? AND _team_filter = ?',
                        (group_size, team_filter)
                    )
                conn.execute('DELETE FROM cache_metadata WHERE cache_name = ?', (cache_name,))
            else:
                # Clear all tables
                for table in ['lineups', 'player_stats', 'team_stats', 'standings']:
                    conn.execute(f'DROP TABLE IF EXISTS {table}')
                conn.execute('DELETE FROM cache_metadata')
            
            conn.commit()
        
        print(f"Cleared cache: {cache_name or 'ALL'}")
    
    def get_db_size(self):
        """Get database file size in MB."""
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path) / (1024 * 1024)
        return 0
    
    def migrate_from_csv(self, data_dir='data'):
        """
        Import existing CSV cache files into the unified SQLite database.
        This is a one-time migration utility.
        
        Args:
            data_dir: Directory containing CSV cache files
        """
        import glob
        
        migrated = []
        
        # 1. Migrate lineup CSVs
        lineup_files = glob.glob(os.path.join(data_dir, 'nba_lineups_cache*.csv'))
        for f in lineup_files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    continue
                
                # Parse filename to get group_size and team
                basename = os.path.basename(f)
                # e.g., nba_lineups_cache_2man.csv or nba_lineups_cache_2man_BOS.csv
                if '2man' in basename:
                    group_size = 2
                elif '3man' in basename:
                    group_size = 3
                else:
                    continue
                
                # Check for team filter
                parts = basename.replace('.csv', '').split('_')
                team_abbr = parts[-1] if len(parts) > 4 and len(parts[-1]) == 3 else None
                
                self.save_lineups(df, group_size, team_abbr, '2024-25')
                migrated.append(f"Lineups: {basename}")
            except Exception as e:
                print(f"Error migrating {f}: {e}")
        
        # 2. Migrate historical stats
        hist_file = os.path.join(data_dir, 'nba_historical_stats.csv')
        if os.path.exists(hist_file):
            try:
                df = pd.read_csv(hist_file)
                if not df.empty:
                    self.save_player_stats(df)
                    migrated.append("Player historical stats")
            except Exception as e:
                print(f"Error migrating historical stats: {e}")
        
        # 3. Migrate team advanced stats
        adv_file = os.path.join(data_dir, 'nba_advanced_stats_cache.csv')
        if os.path.exists(adv_file):
            try:
                df = pd.read_csv(adv_file)
                if not df.empty:
                    self.save_team_stats(df, 'advanced', '2024-25')
                    migrated.append("Team advanced stats")
            except Exception as e:
                print(f"Error migrating team stats: {e}")
        
        # 4. Migrate standings
        standings_file = os.path.join(data_dir, 'nba_standings_cache.csv')
        if os.path.exists(standings_file):
            try:
                df = pd.read_csv(standings_file)
                if not df.empty:
                    self.save_standings(df, '2024-25')
                    migrated.append("Standings")
            except Exception as e:
                print(f"Error migrating standings: {e}")
        
        print(f"\nMigration complete! Imported: {len(migrated)} datasets")
        for item in migrated:
            print(f"  - {item}")
        print(f"\nDatabase size: {self.get_db_size():.2f} MB")
        
        return migrated
    
    def print_status(self):
        """Print a summary of all cached data."""
        print("\n" + "="*60)
        print("SIEVE CACHE STATUS")
        print("="*60)
        print(f"Database: {self.db_path}")
        print(f"Size: {self.get_db_size():.2f} MB")
        print("-"*60)
        
        caches = self.list_caches()
        if caches.empty:
            print("No cached data found.")
        else:
            for _, row in caches.iterrows():
                print(f"\n{row['cache_name']}:")
                print(f"  Records: {row['record_count']}")
                print(f"  Season: {row['season']}")
                print(f"  Updated: {row['last_updated']}")
        
        print("="*60 + "\n")


# Global cache manager instance
cache = CacheManager()

