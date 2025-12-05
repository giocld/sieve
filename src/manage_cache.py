#!/usr/bin/env python3
"""
Cache management CLI for Sieve NBA Analytics.
Run with: python -m src.manage_cache [command]

Database Architecture:
    sieve_players.db - Player stats, contracts, LEBRON metrics, analysis
    sieve_teams.db   - Team stats, standings, lineups, efficiency

Commands:
    status      - Show cache status and contents
    migrate     - Import existing CSV files into split databases
    migrate-old - Migrate from old unified DB to split DBs
    refresh     - Full refresh pipeline: scrape LEBRON -> merge -> recalculate metrics
    clear       - Clear all cached data
    list        - List all cached datasets
    cleanup     - Remove deprecated files after migration
"""

import sys
import os
import argparse
from src.cache_manager import cache
from src.config import CURRENT_SEASON


def cmd_status(args):
    """Show cache status."""
    cache.print_status()


def cmd_migrate(args):
    """Migrate CSV files to SQLite."""
    cache.migrate_from_csv(args.data_dir, season=args.season)


def cmd_migrate_old(args):
    """Migrate from old unified DB to split DBs."""
    cache.migrate_from_old_db(season=args.season)


def cmd_refresh(args):
    """
    Full data refresh pipeline for a specific season.
    
    Steps:
    1. Scrape/load LEBRON data (or use existing CSV)
    2. Merge with contracts
    3. Recalculate player value metrics (value_gap)
    4. Update player_analysis DB
    5. Recalculate team metrics
    6. Update team_efficiency DB
    
    Returns exit code 0 on success, 1 on failure (for cron alerting).
    """
    from src import data_processing
    from src.scraper import scrape_lebron, parse_lebron_csv
    
    season = args.season
    
    print("=" * 70)
    print(f"SIEVE DATA REFRESH PIPELINE")
    print(f"Season: {season}")
    print("=" * 70)
    
    success = True
    
    # Step 1: Get LEBRON data
    print("\n[Step 1/5] Loading LEBRON data...")
    
    df_lebron = None
    
    if args.lebron_csv:
        # Use provided CSV file
        print(f"  Using provided CSV: {args.lebron_csv}")
        df_lebron = parse_lebron_csv(args.lebron_csv, season=season)
    elif args.scrape:
        # Try scraping
        print("  Attempting to scrape from BBall Index...")
        df_lebron = scrape_lebron(season=season, save_csv=True, headless=not args.debug)
    
    if df_lebron is None or df_lebron.empty:
        # Try loading from database
        print("  Trying to load from database...")
        df_lebron = cache.load_lebron_metrics(season=season)
    
    if df_lebron is None or df_lebron.empty:
        print("  ERROR: No LEBRON data available!")
        print("  Options:")
        print(f"    1. Manually download and run: python -m src.manage_cache refresh --season {season} --lebron-csv data/LEBRON.csv")
        print(f"    2. Scrape automatically: python -m src.manage_cache refresh --season {season} --scrape")
        return 1
    
    print(f"  Loaded {len(df_lebron)} LEBRON records")
    
    # Step 2: Load and merge with contracts
    print("\n[Step 2/5] Merging with contracts...")
    try:
        # Determine LEBRON file path for this season
        lebron_file = args.lebron_csv or f'data/LEBRON.csv'
        if not os.path.exists(lebron_file):
            # Try season-specific file
            lebron_file = f'data/LEBRON_{season.replace("-", "_")}.csv'
        
        df_merged = data_processing.load_and_merge_data(
            lebron_file=lebron_file,
            season=season,
            from_db=True
        )
        print(f"  Merged dataset: {len(df_merged)} players")
    except Exception as e:
        print(f"  ERROR: {e}")
        success = False
        df_merged = None
    
    # Step 3: Calculate player value metrics
    if df_merged is not None and not df_merged.empty:
        print("\n[Step 3/5] Calculating player value metrics...")
        try:
            df_analysis = data_processing.calculate_player_value_metrics(df_merged, season=season)
            print(f"  Calculated value_gap for {len(df_analysis)} players")
            
            # Show top/bottom value gap
            if 'value_gap' in df_analysis.columns and 'player_name' in df_analysis.columns:
                top = df_analysis.nlargest(3, 'value_gap')[['player_name', 'value_gap']]
                bot = df_analysis.nsmallest(3, 'value_gap')[['player_name', 'value_gap']]
                print(f"  Top underpaid: {list(top['player_name'])}")
                print(f"  Top overpaid: {list(bot['player_name'])}")
        except Exception as e:
            print(f"  ERROR: {e}")
            success = False
            df_analysis = None
    else:
        df_analysis = None
        success = False
    
    # Step 4: Calculate team metrics
    print("\n[Step 4/5] Calculating team metrics...")
    if df_analysis is not None:
        try:
            df_teams = data_processing.calculate_team_metrics(df_analysis, season=season)
            print(f"  Calculated metrics for {len(df_teams)} teams")
        except Exception as e:
            print(f"  ERROR: {e}")
            success = False
    
    # Step 5: Summary
    print("\n[Step 5/5] Summary")
    print("-" * 40)
    cache.print_status()
    
    if success:
        print("\n" + "=" * 70)
        print("REFRESH COMPLETE - SUCCESS")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("REFRESH COMPLETE - WITH ERRORS")
        print("=" * 70)
        return 1


def cmd_clear(args):
    """Clear cached data."""
    if args.name:
        cache.clear_cache(args.name)
        print(f"Cleared cache: {args.name}")
    elif args.yes:
        cache.clear_cache()
        print("All caches cleared.")
    else:
        confirm = input("Clear ALL cached data? (yes/no): ")
        if confirm.lower() == 'yes':
            cache.clear_cache()
            print("All caches cleared.")
        else:
            print("Cancelled.")


def cmd_list(args):
    """List all caches."""
    caches = cache.list_caches()
    if caches.empty:
        print("No cached data found.")
    else:
        print("\nCached datasets:")
        print("-" * 60)
        for _, row in caches.iterrows():
            print(f"  {row['cache_name']}: {row['record_count']} records (updated: {row['last_updated'][:19]})")


def cmd_cleanup(args):
    """List and optionally remove deprecated files after migration."""
    import os
    
    # Files that can be safely removed after migration
    deprecated_files = [
        # Old CSV cache files
        'data/basketball_reference_contracts.csv',
        'data/nba_historical_stats.csv',
        'data/nba_advanced_stats_cache.csv',
        'data/nba_standings_cache.csv',
        'data/nba_lineups_cache_2man.csv',
        'data/nba_lineups_cache_3man.csv',
        'data/sieve_analysis.csv',
        'data/sieve_player_analysis.csv',
        'data/sieve_team_efficiency.csv',
        # Old unified database (after migrating to split DBs)
        'data/sieve_cache.db',
    ]
    
    # Files to keep
    keep_files = [
        # External raw inputs
        'data/LEBRON.csv',
        'data/bbref_contracts_raw.csv',
        'data/nba_teams.json',
        # New split databases
        'data/sieve_players.db',
        'data/sieve_teams.db',
    ]
    
    print("="*60)
    print("CSV CLEANUP UTILITY")
    print("="*60)
    
    # Check migration status first
    caches = cache.list_caches()
    if caches.empty:
        print("\nWARNING: No data found in database!")
        print("Run 'python -m src.manage_cache migrate' first before cleanup.")
        return
    
    print(f"\nDatabase contains {len(caches)} cached datasets.")
    
    # Find existing deprecated files
    existing_deprecated = []
    for f in deprecated_files:
        if os.path.exists(f):
            size_kb = os.path.getsize(f) / 1024
            existing_deprecated.append((f, size_kb))
    
    if not existing_deprecated:
        print("\nNo deprecated CSV files found. Cleanup complete!")
        return
    
    print(f"\nDeprecated CSV files that can be removed ({len(existing_deprecated)} files):")
    print("-"*60)
    total_size = 0
    for f, size in existing_deprecated:
        print(f"  [X] {f} ({size:.1f} KB)")
        total_size += size
    print(f"\nTotal space to free: {total_size:.1f} KB ({total_size/1024:.2f} MB)")
    
    print(f"\nFiles to KEEP (raw inputs):")
    print("-"*60)
    for f in keep_files:
        status = "[OK]" if os.path.exists(f) else "[--]"
        print(f"  {status} {f}")
    
    if args.delete:
        print("\n" + "="*60)
        if args.yes:
            confirm = 'yes'
        else:
            confirm = input("DELETE all deprecated files? This cannot be undone! (yes/no): ")
        if confirm.lower() == 'yes':
            deleted = 0
            for f, _ in existing_deprecated:
                try:
                    os.remove(f)
                    print(f"  Deleted: {f}")
                    deleted += 1
                except Exception as e:
                    print(f"  Error deleting {f}: {e}")
            print(f"\nDeleted {deleted} files.")
        else:
            print("Cleanup cancelled.")
    else:
        print("\nTo delete these files, run:")
        print("  python -m src.manage_cache cleanup --delete")


def main():
    parser = argparse.ArgumentParser(
        description="Sieve Cache Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show cache status')
    status_parser.set_defaults(func=cmd_status)
    
    # Migrate command (from CSVs)
    migrate_parser = subparsers.add_parser('migrate', help='Migrate CSVs to split databases')
    migrate_parser.add_argument('--data-dir', default='data', help='Directory with CSV files')
    migrate_parser.add_argument('--season', default='2024-25', help='NBA season string (e.g., 2024-25)')
    migrate_parser.set_defaults(func=cmd_migrate)
    
    # Migrate-old command (from old unified DB)
    migrate_old_parser = subparsers.add_parser('migrate-old', help='Migrate from old unified DB to split DBs')
    migrate_old_parser.add_argument('--season', default='2024-25', help='NBA season string')
    migrate_old_parser.set_defaults(func=cmd_migrate_old)
    
    # Refresh command (full pipeline)
    refresh_parser = subparsers.add_parser('refresh', help='Full refresh: LEBRON -> merge -> metrics')
    refresh_parser.add_argument('--season', default=CURRENT_SEASON, help=f'NBA season (default: {CURRENT_SEASON})')
    refresh_parser.add_argument('--lebron-csv', help='Path to LEBRON CSV file (if not scraping)')
    refresh_parser.add_argument('--scrape', action='store_true', help='Attempt to scrape LEBRON data')
    refresh_parser.add_argument('--debug', action='store_true', help='Show browser (non-headless) for debugging')
    refresh_parser.set_defaults(func=cmd_refresh)
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cached data')
    clear_parser.add_argument('--name', help='Specific cache name to clear')
    clear_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    clear_parser.set_defaults(func=cmd_clear)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all caches')
    list_parser.set_defaults(func=cmd_list)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove deprecated CSV files after migration')
    cleanup_parser.add_argument('--delete', action='store_true', help='Actually delete the files (otherwise just list)')
    cleanup_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()

