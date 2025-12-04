#!/usr/bin/env python3
"""
Cache management CLI for Sieve NBA Analytics.
Run with: python -m src.manage_cache [command]

Commands:
    status   - Show cache status and contents
    migrate  - Import existing CSV files into unified SQLite database
    clear    - Clear all cached data
    refresh  - Clear and refetch all data
"""

import sys
import argparse
from src.cache_manager import cache


def cmd_status(args):
    """Show cache status."""
    cache.print_status()


def cmd_migrate(args):
    """Migrate CSV files to SQLite."""
    print("Migrating existing CSV cache files to SQLite database...")
    print("-" * 60)
    cache.migrate_from_csv(args.data_dir)


def cmd_clear(args):
    """Clear cached data."""
    if args.name:
        cache.clear_cache(args.name)
        print(f"Cleared cache: {args.name}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Sieve Cache Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show cache status')
    status_parser.set_defaults(func=cmd_status)
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate CSVs to SQLite')
    migrate_parser.add_argument('--data-dir', default='data', help='Directory with CSV files')
    migrate_parser.set_defaults(func=cmd_migrate)
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cached data')
    clear_parser.add_argument('--name', help='Specific cache name to clear')
    clear_parser.set_defaults(func=cmd_clear)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all caches')
    list_parser.set_defaults(func=cmd_list)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()

