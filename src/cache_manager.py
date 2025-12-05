"""
Cache Manager for Sieve NBA Analytics.
Provides SQLite-based caching with separate databases for player and team data.

Database Architecture:
    sieve_players.db - Player-related data
        - player_stats: Historical player statistics
        - lebron_metrics: LEBRON impact data
        - contracts: Player contract data
        - player_analysis: Value gap analysis
        
    sieve_teams.db - Team-related data
        - team_stats: Team advanced stats
        - standings: Team standings/wins
        - team_efficiency: Team efficiency rankings
        - lineups: Duo/trio lineup data
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
import json


class BaseCacheManager:
    """Base class with common database operations."""
    
    def __init__(self, db_path):
        """Initialize with database path."""
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database and metadata table if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
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
    
    def _table_exists(self, conn, table_name):
        """Check if a table exists in the database."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None
    
    def get_cache_info(self, cache_name):
        """Get metadata about a specific cache."""
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
    
    def is_cache_fresh(self, cache_name, max_age_hours=24):
        """Check if a cache is fresh enough to use."""
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
    
    def get_db_size(self):
        """Get database file size in MB."""
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path) / (1024 * 1024)
        return 0


class PlayerCacheManager(BaseCacheManager):
    """
    Manages player-related cached data.
    
    Tables:
        - player_stats: Historical player statistics (from NBA API)
        - lebron_metrics: LEBRON impact data (from external CSV)
        - contracts: Player contract data (from BBRef)
        - player_analysis: Value gap analysis (computed)
    """
    
    def __init__(self, db_path='data/sieve_players.db'):
        super().__init__(db_path)
    
    # =========================================================================
    # PLAYER HISTORICAL STATS
    # =========================================================================
    
    def save_player_stats(self, df, season=None):
        """Save historical player stats to cache."""
        if df.empty:
            return
        
        cache_name = 'player_stats'
        
        with self._get_connection() as conn:
            if season:
                # Single season update
                if self._table_exists(conn, 'player_stats'):
                    conn.execute('DELETE FROM player_stats WHERE SEASON_ID = ?', (season,))
                df.to_sql('player_stats', conn, if_exists='append', index=False)
            else:
                # Full bulk save
                df.to_sql('player_stats', conn, if_exists='replace', index=False)
            
            count = conn.execute('SELECT COUNT(*) FROM player_stats').fetchone()[0]
            self._update_metadata(conn, cache_name, season, count)
            conn.commit()
        
        print(f"[Players DB] Cached {len(df)} player stat records")
    
    def load_player_stats(self, seasons=None, min_games=0):
        """Load historical player stats from cache."""
        try:
            with self._get_connection() as conn:
                if not self._table_exists(conn, 'player_stats'):
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
    # LEBRON METRICS
    # =========================================================================
    
    def save_lebron_metrics(self, df, season='2024-25'):
        """Save LEBRON player impact metrics to cache.
        
        Uses DELETE + append pattern to preserve other seasons' data.
        """
        if df.empty:
            return
        
        df = df.copy()
        df['_season'] = season
        
        with self._get_connection() as conn:
            # Delete only this season's data, then append (preserves other seasons)
            if self._table_exists(conn, 'lebron_metrics'):
                conn.execute('DELETE FROM lebron_metrics WHERE _season = ?', (season,))
            
            df.to_sql('lebron_metrics', conn, if_exists='append', index=False)
            self._update_metadata(conn, 'lebron_metrics', season, len(df))
            conn.commit()
        
        print(f"[Players DB] Cached {len(df)} LEBRON metric records (season: {season})")
    
    def load_lebron_metrics(self, season=None):
        """Load LEBRON metrics from cache.
        
        Args:
            season: Specific season to load (e.g., '2024-25'), or None to load all seasons.
            
        Returns:
            DataFrame with LEBRON metrics, or None if not found.
        """
        try:
            with self._get_connection() as conn:
                if not self._table_exists(conn, 'lebron_metrics'):
                    return None
                
                if season:
                    df = pd.read_sql(
                        'SELECT * FROM lebron_metrics WHERE _season = ?',
                        conn, params=[season]
                    )
                else:
                    # Load all seasons
                    df = pd.read_sql('SELECT * FROM lebron_metrics', conn)
                
                if df.empty:
                    return None
                
                return df
                
        except Exception as e:
            print(f"Error loading LEBRON metrics: {e}")
            return None
    
    def list_lebron_seasons(self):
        """List all seasons with LEBRON data."""
        try:
            with self._get_connection() as conn:
                if not self._table_exists(conn, 'lebron_metrics'):
                    return []
                
                cursor = conn.execute('SELECT DISTINCT _season FROM lebron_metrics')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error listing LEBRON seasons: {e}")
            return []
    
    # =========================================================================
    # CONTRACTS
    # =========================================================================
    
    def save_contracts(self, df, season='2024-25'):
        """Save processed contract data to cache.
        
        Uses DELETE + append pattern to preserve other seasons' data.
        """
        if df.empty:
            return
        
        df = df.copy()
        df['_season'] = season
        
        with self._get_connection() as conn:
            # Delete only this season's data, then append (preserves other seasons)
            if self._table_exists(conn, 'contracts'):
                conn.execute('DELETE FROM contracts WHERE _season = ?', (season,))
            
            df.to_sql('contracts', conn, if_exists='append', index=False)
            self._update_metadata(conn, 'contracts', season, len(df))
            conn.commit()
        
        print(f"[Players DB] Cached {len(df)} contract records (season: {season})")
    
    def load_contracts(self, season='2024-25'):
        """Load contract data from cache."""
        try:
            with self._get_connection() as conn:
                if not self._table_exists(conn, 'contracts'):
                    return None
                
                df = pd.read_sql(
                    'SELECT * FROM contracts WHERE _season = ?',
                    conn, params=[season]
                )
                
                if df.empty:
                    return None
                
                df = df.drop(columns=['_season'], errors='ignore')
                return df
                
        except Exception as e:
            print(f"Error loading contracts: {e}")
            return None
    
    # =========================================================================
    # PLAYER ANALYSIS (Value Gap)
    # =========================================================================
    
    def save_player_analysis(self, df, season='2024-25'):
        """Save player value analysis to cache.
        
        Uses DELETE + append pattern to preserve other seasons' data.
        Handles schema changes by recreating the table if needed.
        """
        if df.empty:
            return
        
        df = df.copy()
        df['_season'] = season
        
        with self._get_connection() as conn:
            if self._table_exists(conn, 'player_analysis'):
                # Check if schema matches by comparing columns
                cursor = conn.execute("PRAGMA table_info(player_analysis)")
                existing_cols = set(row[1] for row in cursor.fetchall())
                new_cols = set(df.columns)
                
                # If schemas differ significantly, recreate table
                if not new_cols.issubset(existing_cols):
                    print(f"  Schema change detected, recreating player_analysis table...")
                    conn.execute('DROP TABLE player_analysis')
                else:
                    conn.execute('DELETE FROM player_analysis WHERE _season = ?', (season,))
            
            df.to_sql('player_analysis', conn, if_exists='append', index=False)
            self._update_metadata(conn, 'player_analysis', season, len(df))
            conn.commit()
        
        print(f"[Players DB] Cached {len(df)} player analysis records (season: {season})")
    
    def load_player_analysis(self, season='2024-25'):
        """Load player analysis from cache."""
        try:
            with self._get_connection() as conn:
                if not self._table_exists(conn, 'player_analysis'):
                    return None
                
                df = pd.read_sql(
                    'SELECT * FROM player_analysis WHERE _season = ?',
                    conn, params=[season]
                )
                
                if df.empty:
                    return None
                
                df = df.drop(columns=['_season'], errors='ignore')
                return df
                
        except Exception as e:
            print(f"Error loading player analysis: {e}")
            return None
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def clear_all(self):
        """Clear all player data."""
        with self._get_connection() as conn:
            for table in ['player_stats', 'lebron_metrics', 'contracts', 'player_analysis']:
                conn.execute(f'DROP TABLE IF EXISTS {table}')
            conn.execute('DELETE FROM cache_metadata')
            conn.commit()
        print(f"[Players DB] Cleared all data")


class TeamCacheManager(BaseCacheManager):
    """
    Manages team-related cached data.
    
    Tables:
        - team_stats: Team advanced stats (from NBA API)
        - standings: Team standings/wins (from NBA API)
        - team_efficiency: Team efficiency rankings (computed)
        - lineups: Duo/trio lineup data (from NBA API)
    """
    
    def __init__(self, db_path='data/sieve_teams.db'):
        super().__init__(db_path)
    
    # =========================================================================
    # TEAM STATS
    # =========================================================================
    
    def save_team_stats(self, df, stat_type='advanced', season='2024-25'):
        """Save team stats to cache.
        
        Handles schema changes by recreating table if needed.
        """
        if df.empty:
            return
        
        df = df.copy()
        df['_stat_type'] = stat_type
        df['_season'] = season
        
        cache_name = f'team_stats_{stat_type}'
        
        with self._get_connection() as conn:
            if self._table_exists(conn, 'team_stats'):
                # Check schema compatibility
                cursor = conn.execute("PRAGMA table_info(team_stats)")
                existing_cols = set(row[1] for row in cursor.fetchall())
                new_cols = set(df.columns)
                
                if not new_cols.issubset(existing_cols):
                    print(f"  Schema change detected, recreating team_stats table...")
                    conn.execute('DROP TABLE team_stats')
                else:
                    conn.execute(
                        'DELETE FROM team_stats WHERE _stat_type = ? AND _season = ?',
                        (stat_type, season)
                    )
            
            df.to_sql('team_stats', conn, if_exists='append', index=False)
            self._update_metadata(conn, cache_name, season, len(df))
            conn.commit()
        
        print(f"[Teams DB] Cached {len(df)} team {stat_type} records")
    
    def load_team_stats(self, stat_type='advanced', season='2024-25'):
        """Load team stats from cache."""
        try:
            with self._get_connection() as conn:
                if not self._table_exists(conn, 'team_stats'):
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
        """Save standings to cache.
        
        Handles schema changes by recreating table if needed.
        """
        if df.empty:
            return
        
        df = df.copy()
        df['_season'] = season
        
        with self._get_connection() as conn:
            if self._table_exists(conn, 'standings'):
                # Check schema compatibility
                cursor = conn.execute("PRAGMA table_info(standings)")
                existing_cols = set(row[1] for row in cursor.fetchall())
                new_cols = set(df.columns)
                
                if not new_cols.issubset(existing_cols):
                    print(f"  Schema change detected, recreating standings table...")
                    conn.execute('DROP TABLE standings')
                else:
                    conn.execute('DELETE FROM standings WHERE _season = ?', (season,))
            
            df.to_sql('standings', conn, if_exists='append', index=False)
            self._update_metadata(conn, 'standings', season, len(df))
            conn.commit()
        
        print(f"[Teams DB] Cached {len(df)} team standings (season: {season})")
    
    def load_standings(self, season='2024-25'):
        """Load standings from cache."""
        try:
            with self._get_connection() as conn:
                if not self._table_exists(conn, 'standings'):
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
    # TEAM EFFICIENCY
    # =========================================================================
    
    def save_team_efficiency(self, df, season='2024-25'):
        """Save team efficiency rankings to cache."""
        if df.empty:
            return
        
        df = df.copy()
        df['_season'] = season
        
        with self._get_connection() as conn:
            if self._table_exists(conn, 'team_efficiency'):
                conn.execute('DELETE FROM team_efficiency WHERE _season = ?', (season,))
            
            df.to_sql('team_efficiency', conn, if_exists='append', index=False)
            self._update_metadata(conn, 'team_efficiency', season, len(df))
            conn.commit()
        
        print(f"[Teams DB] Cached {len(df)} team efficiency records")
    
    def load_team_efficiency(self, season='2024-25'):
        """Load team efficiency rankings from cache."""
        try:
            with self._get_connection() as conn:
                if not self._table_exists(conn, 'team_efficiency'):
                    return None
                
                df = pd.read_sql(
                    'SELECT * FROM team_efficiency WHERE _season = ?',
                    conn, params=[season]
                )
                
                if df.empty:
                    return None
                
                df = df.drop(columns=['_season'], errors='ignore')
                return df
                
        except Exception as e:
            print(f"Error loading team efficiency: {e}")
            return None
    
    # =========================================================================
    # LINEUPS
    # =========================================================================
    
    def save_lineups(self, df, group_size, team_abbr=None, season='2024-25'):
        """Save lineup data to cache."""
        if df.empty:
            return
        
        df = df.copy()
        df['_group_size'] = group_size
        df['_team_filter'] = team_abbr or 'ALL'
        df['_season'] = season
        
        cache_name = f'lineups_{group_size}man_{team_abbr or "ALL"}'
        
        with self._get_connection() as conn:
            if self._table_exists(conn, 'lineups'):
                conn.execute('''
                    DELETE FROM lineups 
                    WHERE _group_size = ? AND _team_filter = ? AND _season = ?
                ''', (group_size, team_abbr or 'ALL', season))
            
            df.to_sql('lineups', conn, if_exists='append', index=False)
            self._update_metadata(conn, cache_name, season, len(df), {
                'group_size': group_size,
                'team_abbr': team_abbr
            })
            conn.commit()
        
        print(f"[Teams DB] Cached {len(df)} lineups to {cache_name}")
    
    def load_lineups(self, group_size, team_abbr=None, season='2024-25', min_minutes=0):
        """Load lineup data from cache."""
        cache_name = f'lineups_{group_size}man_{team_abbr or "ALL"}'
        
        try:
            with self._get_connection() as conn:
                if not self._table_exists(conn, 'lineups'):
                    return None
                
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
                
                # Drop internal columns
                internal_cols = ['_group_size', '_team_filter', '_season']
                df = df.drop(columns=[c for c in internal_cols if c in df.columns], errors='ignore')
                
                print(f"[Teams DB] Loaded {len(df)} lineups from cache: {cache_name}")
                return df
                
        except Exception as e:
            print(f"Cache read error: {e}")
            return None
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def clear_all(self):
        """Clear all team data."""
        with self._get_connection() as conn:
            for table in ['team_stats', 'standings', 'team_efficiency', 'lineups']:
                conn.execute(f'DROP TABLE IF EXISTS {table}')
            conn.execute('DELETE FROM cache_metadata')
            conn.commit()
        print(f"[Teams DB] Cleared all data")


class CacheManager:
    """
    Unified cache manager that delegates to player and team databases.
    
    This provides a single interface for all caching operations while
    keeping player and team data in separate databases for clarity.
    """
    
    def __init__(self, players_db='data/sieve_players.db', teams_db='data/sieve_teams.db'):
        """Initialize with paths to both databases."""
        self.players = PlayerCacheManager(players_db)
        self.teams = TeamCacheManager(teams_db)
        
        # Store paths for status reporting
        self.players_db = players_db
        self.teams_db = teams_db
    
    # =========================================================================
    # PLAYER DATA (delegated to PlayerCacheManager)
    # =========================================================================
    
    def save_player_stats(self, df, season=None):
        return self.players.save_player_stats(df, season)
    
    def load_player_stats(self, seasons=None, min_games=0):
        return self.players.load_player_stats(seasons, min_games)
    
    def save_lebron_metrics(self, df, season='2024-25'):
        return self.players.save_lebron_metrics(df, season)
    
    def load_lebron_metrics(self, season=None):
        return self.players.load_lebron_metrics(season)
    
    def list_lebron_seasons(self):
        return self.players.list_lebron_seasons()
    
    def save_contracts(self, df, season='2024-25'):
        return self.players.save_contracts(df, season)
    
    def load_contracts(self, season='2024-25'):
        return self.players.load_contracts(season)
    
    def save_player_analysis(self, df, season='2024-25'):
        return self.players.save_player_analysis(df, season)
    
    def load_player_analysis(self, season='2024-25'):
        return self.players.load_player_analysis(season)
    
    # =========================================================================
    # TEAM DATA (delegated to TeamCacheManager)
    # =========================================================================
    
    def save_team_stats(self, df, stat_type='advanced', season='2024-25'):
        return self.teams.save_team_stats(df, stat_type, season)
    
    def load_team_stats(self, stat_type='advanced', season='2024-25'):
        return self.teams.load_team_stats(stat_type, season)
    
    def save_standings(self, df, season='2024-25'):
        return self.teams.save_standings(df, season)
    
    def load_standings(self, season='2024-25'):
        return self.teams.load_standings(season)
    
    def save_team_efficiency(self, df, season='2024-25'):
        return self.teams.save_team_efficiency(df, season)
    
    def load_team_efficiency(self, season='2024-25'):
        return self.teams.load_team_efficiency(season)
    
    def save_lineups(self, df, group_size, team_abbr=None, season='2024-25'):
        return self.teams.save_lineups(df, group_size, team_abbr, season)
    
    def load_lineups(self, group_size, team_abbr=None, season='2024-25', min_minutes=0):
        return self.teams.load_lineups(group_size, team_abbr, season, min_minutes)
    
    # =========================================================================
    # UNIFIED OPERATIONS
    # =========================================================================
    
    def clear_cache(self, cache_name=None):
        """Clear cached data from appropriate database."""
        if cache_name:
            # Determine which DB based on cache name
            player_caches = ['player_stats', 'lebron_metrics', 'contracts', 'player_analysis']
            if any(cache_name.startswith(p) for p in player_caches):
                # Clear from players DB (simplified - clear table)
                print(f"Clearing {cache_name} from players DB")
            else:
                # Clear from teams DB
                print(f"Clearing {cache_name} from teams DB")
        else:
            self.players.clear_all()
            self.teams.clear_all()
    
    def list_caches(self):
        """List all caches from both databases."""
        players_caches = self.players.list_caches()
        players_caches['database'] = 'players'
        
        teams_caches = self.teams.list_caches()
        teams_caches['database'] = 'teams'
        
        return pd.concat([players_caches, teams_caches], ignore_index=True)
    
    def get_cache_info(self, cache_name):
        """Get cache info, checking both databases."""
        info = self.players.get_cache_info(cache_name)
        if info:
            return info
        return self.teams.get_cache_info(cache_name)
    
    def is_cache_fresh(self, cache_name, max_age_hours=24):
        """Check if cache is fresh in either database."""
        return (self.players.is_cache_fresh(cache_name, max_age_hours) or 
                self.teams.is_cache_fresh(cache_name, max_age_hours))
    
    def migrate_from_csv(self, data_dir='data', season='2024-25'):
        """
        Migrate existing CSV files to the split database architecture.
        """
        import glob
        
        migrated = []
        
        print("="*60)
        print("SIEVE CSV TO DATABASE MIGRATION (Split Architecture)")
        print("="*60)
        print(f"\nPlayers DB: {self.players_db}")
        print(f"Teams DB:   {self.teams_db}")
        print("-"*60)
        
        # === PLAYER DATA ===
        print("\n[PLAYERS DATABASE]")
        
        # Historical stats
        hist_file = os.path.join(data_dir, 'nba_historical_stats.csv')
        if os.path.exists(hist_file):
            try:
                df = pd.read_csv(hist_file)
                if not df.empty:
                    self.save_player_stats(df)
                    migrated.append(f"Player stats ({len(df)} records)")
            except Exception as e:
                print(f"Error migrating historical stats: {e}")
        
        # LEBRON metrics
        lebron_file = os.path.join(data_dir, 'LEBRON.csv')
        if os.path.exists(lebron_file):
            try:
                df = pd.read_csv(lebron_file)
                if not df.empty:
                    self.save_lebron_metrics(df, season)
                    migrated.append(f"LEBRON metrics ({len(df)} players)")
            except Exception as e:
                print(f"Error migrating LEBRON: {e}")
        
        # Contracts
        contracts_file = os.path.join(data_dir, 'basketball_reference_contracts.csv')
        if os.path.exists(contracts_file):
            try:
                df = pd.read_csv(contracts_file)
                if not df.empty:
                    self.save_contracts(df, season)
                    migrated.append(f"Contracts ({len(df)} players)")
            except Exception as e:
                print(f"Error migrating contracts: {e}")
        
        # Player analysis
        analysis_file = os.path.join(data_dir, 'sieve_player_analysis.csv')
        if os.path.exists(analysis_file):
            try:
                df = pd.read_csv(analysis_file)
                if not df.empty:
                    self.save_player_analysis(df, season)
                    migrated.append(f"Player analysis ({len(df)} records)")
            except Exception as e:
                print(f"Error migrating player analysis: {e}")
        
        # === TEAM DATA ===
        print("\n[TEAMS DATABASE]")
        
        # Team advanced stats
        adv_file = os.path.join(data_dir, 'nba_advanced_stats_cache.csv')
        if os.path.exists(adv_file):
            try:
                df = pd.read_csv(adv_file)
                if not df.empty:
                    self.save_team_stats(df, 'advanced', season)
                    migrated.append("Team advanced stats")
            except Exception as e:
                print(f"Error migrating team stats: {e}")
        
        # Standings
        standings_file = os.path.join(data_dir, 'nba_standings_cache.csv')
        if os.path.exists(standings_file):
            try:
                df = pd.read_csv(standings_file)
                if not df.empty:
                    self.save_standings(df, season)
                    migrated.append("Standings")
            except Exception as e:
                print(f"Error migrating standings: {e}")
        
        # Team efficiency
        efficiency_file = os.path.join(data_dir, 'sieve_team_efficiency.csv')
        if os.path.exists(efficiency_file):
            try:
                df = pd.read_csv(efficiency_file)
                if not df.empty:
                    self.save_team_efficiency(df, season)
                    migrated.append(f"Team efficiency ({len(df)} teams)")
            except Exception as e:
                print(f"Error migrating team efficiency: {e}")
        
        # Lineups
        lineup_files = glob.glob(os.path.join(data_dir, 'nba_lineups_cache*.csv'))
        for f in lineup_files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    continue
                
                basename = os.path.basename(f)
                if '2man' in basename:
                    group_size = 2
                elif '3man' in basename:
                    group_size = 3
                else:
                    continue
                
                parts = basename.replace('.csv', '').split('_')
                team_abbr = parts[-1] if len(parts) > 4 and len(parts[-1]) == 3 else None
                
                self.save_lineups(df, group_size, team_abbr, season)
                migrated.append(f"Lineups: {basename}")
            except Exception as e:
                print(f"Error migrating {f}: {e}")
        
        # Summary
        print("\n" + "="*60)
        print(f"MIGRATION COMPLETE - {len(migrated)} datasets migrated")
        print("="*60)
        for item in migrated:
            print(f"  [OK] {item}")
        
        print(f"\nDatabase sizes:")
        print(f"  Players: {self.players.get_db_size():.2f} MB")
        print(f"  Teams:   {self.teams.get_db_size():.2f} MB")
        print("="*60)
        
        return migrated
    
    def migrate_from_old_db(self, old_db_path='data/sieve_cache.db', season='2024-25'):
        """
        Migrate from the old unified database to the new split architecture.
        """
        if not os.path.exists(old_db_path):
            print(f"Old database not found: {old_db_path}")
            return []
        
        migrated = []
        
        print("="*60)
        print("MIGRATING FROM OLD UNIFIED DATABASE")
        print("="*60)
        print(f"\nSource: {old_db_path}")
        print(f"Target Players DB: {self.players_db}")
        print(f"Target Teams DB:   {self.teams_db}")
        print("-"*60)
        
        with sqlite3.connect(old_db_path) as old_conn:
            # Check what tables exist
            cursor = old_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            print(f"\nFound tables: {tables}")
            
            # === PLAYER DATA ===
            print("\n[MIGRATING PLAYER DATA]")
            
            if 'player_stats' in tables:
                df = pd.read_sql('SELECT * FROM player_stats', old_conn)
                if not df.empty:
                    self.save_player_stats(df)
                    migrated.append(f"player_stats ({len(df)} records)")
            
            if 'lebron_metrics' in tables:
                df = pd.read_sql('SELECT * FROM lebron_metrics', old_conn)
                if not df.empty:
                    # Remove _season column if present (will be re-added)
                    df = df.drop(columns=['_season'], errors='ignore')
                    self.save_lebron_metrics(df, season)
                    migrated.append(f"lebron_metrics ({len(df)} records)")
            
            if 'contracts' in tables:
                df = pd.read_sql('SELECT * FROM contracts', old_conn)
                if not df.empty:
                    df = df.drop(columns=['_season'], errors='ignore')
                    self.save_contracts(df, season)
                    migrated.append(f"contracts ({len(df)} records)")
            
            if 'player_analysis' in tables:
                df = pd.read_sql('SELECT * FROM player_analysis', old_conn)
                if not df.empty:
                    df = df.drop(columns=['_season'], errors='ignore')
                    self.save_player_analysis(df, season)
                    migrated.append(f"player_analysis ({len(df)} records)")
            
            # === TEAM DATA ===
            print("\n[MIGRATING TEAM DATA]")
            
            if 'team_stats' in tables:
                df = pd.read_sql('SELECT * FROM team_stats', old_conn)
                if not df.empty:
                    # Get stat_type from the data
                    for stat_type in df['_stat_type'].unique() if '_stat_type' in df.columns else ['advanced']:
                        subset = df[df['_stat_type'] == stat_type] if '_stat_type' in df.columns else df
                        subset = subset.drop(columns=['_stat_type', '_season'], errors='ignore')
                        self.save_team_stats(subset, stat_type, season)
                    migrated.append(f"team_stats ({len(df)} records)")
            
            if 'standings' in tables:
                df = pd.read_sql('SELECT * FROM standings', old_conn)
                if not df.empty:
                    df = df.drop(columns=['_season'], errors='ignore')
                    self.save_standings(df, season)
                    migrated.append(f"standings ({len(df)} records)")
            
            if 'team_efficiency' in tables:
                df = pd.read_sql('SELECT * FROM team_efficiency', old_conn)
                if not df.empty:
                    df = df.drop(columns=['_season'], errors='ignore')
                    self.save_team_efficiency(df, season)
                    migrated.append(f"team_efficiency ({len(df)} records)")
            
            if 'lineups' in tables:
                df = pd.read_sql('SELECT * FROM lineups', old_conn)
                if not df.empty:
                    # Group by group_size and team_filter
                    for (gs, tf, s), group in df.groupby(['_group_size', '_team_filter', '_season']):
                        group = group.drop(columns=['_group_size', '_team_filter', '_season'], errors='ignore')
                        team = tf if tf != 'ALL' else None
                        self.save_lineups(group, int(gs), team, s)
                    migrated.append(f"lineups ({len(df)} records)")
        
        # Summary
        print("\n" + "="*60)
        print(f"MIGRATION COMPLETE - {len(migrated)} tables migrated")
        print("="*60)
        for item in migrated:
            print(f"  [OK] {item}")
        
        print(f"\nDatabase sizes:")
        print(f"  Players: {self.players.get_db_size():.2f} MB")
        print(f"  Teams:   {self.teams.get_db_size():.2f} MB")
        print("="*60)
        
        return migrated
    
    def print_status(self):
        """Print status of both databases."""
        print("\n" + "="*60)
        print("SIEVE CACHE STATUS (Split Database Architecture)")
        print("="*60)
        
        # Players DB
        print(f"\n[PLAYERS DATABASE]")
        print(f"Path: {self.players_db}")
        print(f"Size: {self.players.get_db_size():.2f} MB")
        print("-"*40)
        
        players_caches = self.players.list_caches()
        if players_caches.empty:
            print("  No cached data.")
        else:
            for _, row in players_caches.iterrows():
                print(f"  {row['cache_name']}: {row['record_count']} records")
        
        # Teams DB
        print(f"\n[TEAMS DATABASE]")
        print(f"Path: {self.teams_db}")
        print(f"Size: {self.teams.get_db_size():.2f} MB")
        print("-"*40)
        
        teams_caches = self.teams.list_caches()
        if teams_caches.empty:
            print("  No cached data.")
        else:
            for _, row in teams_caches.iterrows():
                print(f"  {row['cache_name']}: {row['record_count']} records")
        
        total_size = self.players.get_db_size() + self.teams.get_db_size()
        print(f"\nTotal size: {total_size:.2f} MB")
        print("="*60 + "\n")


# Global cache manager instance
cache = CacheManager()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'migrate':
            season = sys.argv[2] if len(sys.argv) > 2 else '2024-25'
            cache.migrate_from_csv(data_dir='data', season=season)
            
        elif command == 'migrate-old':
            # Migrate from old unified DB to new split DBs
            season = sys.argv[2] if len(sys.argv) > 2 else '2024-25'
            cache.migrate_from_old_db(season=season)
            
        elif command == 'status':
            cache.print_status()
            
        elif command == 'clear':
            confirm = input("Clear ALL caches in both databases? (yes/no): ")
            if confirm.lower() == 'yes':
                cache.clear_cache()
                print("All caches cleared.")
            else:
                print("Cancelled.")
                
        else:
            print("Unknown command. Available commands:")
            print("  migrate [season]     - Migrate CSV files to databases")
            print("  migrate-old [season] - Migrate from old unified DB")
            print("  status               - Show cache status")
            print("  clear                - Clear all caches")
    else:
        print("Sieve Cache Manager (Split Database Architecture)")
        print("-" * 50)
        print("Usage: python -m src.cache_manager <command>")
        print("")
        print("Commands:")
        print("  migrate [season]     - Migrate CSV files to databases")
        print("  migrate-old [season] - Migrate from old unified DB")
        print("  status               - Show cache status")
        print("  clear                - Clear all caches")
        print("")
        cache.print_status()
