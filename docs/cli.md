# CLI Reference

Command-line tools for managing Sieve data.

## manage_cache.py

Cache management CLI.

### Usage

```bash
python -m src.manage_cache <command> [options]
```

### Commands

#### status

Show cache status for all databases.

```bash
python -m src.manage_cache status
```

#### clear

Clear all cache data (or specific season).

```bash
python -m src.manage_cache clear --yes
python -m src.manage_cache clear --season 2024-25 --yes
```

Options:
- `--season`: Only clear data for this season
- `--yes, -y`: Skip confirmation prompt

#### migrate

Migrate data from CSV files to database.

```bash
python -m src.manage_cache migrate
```

#### refresh

Run full data pipeline for a season.

```bash
python -m src.manage_cache refresh
python -m src.manage_cache refresh --season 2025-26 --lebron-csv data/LEBRON_2025_26.csv
```

Options:
- `--season`: Season to refresh (default: CURRENT_SEASON)
- `--lebron-csv`: Path to LEBRON CSV file

#### cleanup

Remove stale or unused data.

```bash
python -m src.manage_cache cleanup
python -m src.manage_cache cleanup --delete --yes
```

## scraper.py

Data scraping CLI.

### Usage

```bash
python -m src.scraper <options>
```

### Options

#### --contracts

Scrape contract data from Basketball Reference.

```bash
python -m src.scraper --contracts
```

#### --lebron

Scrape LEBRON data from BBall Index.

```bash
python -m src.scraper --lebron --season 2025-26
```

#### --lebron-csv

Process a manually downloaded LEBRON CSV.

```bash
python -m src.scraper --lebron-csv path/to/file.csv --season 2025-26
```

## Workflow Example

```bash
# 1. Scrape contracts
python -m src.scraper --contracts

# 2. Process LEBRON CSV
python -m src.scraper --lebron-csv data/lebron_new.csv --season 2025-26

# 3. Run full refresh
python -m src.manage_cache refresh --season 2025-26
```

