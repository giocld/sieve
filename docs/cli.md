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
python -m src.manage_cache refresh --season 2025-26 --force
```

Options:
- `--season`: Season to refresh (default: CURRENT_SEASON)
- `--lebron-csv`: Path to LEBRON CSV file
- `--force`: Force re-fetch of NBA API data (standings, per-game stats, team
  advanced) instead of serving from the database cache. Use this when the
  upstream NBA data has changed (new games played, rosters updated) but the
  pipeline would otherwise reuse the stored snapshot.

> **Note:** After a refresh, the running API server still holds the previous
> snapshot in its in-memory caches. Hit `POST /api/admin/reload` (see below) or
> restart the server so clients see the new data.

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

## HTTP Admin Endpoints

The FastAPI backend caches computed season data and ML models in memory for
performance. After re-running the pipeline you need to invalidate these caches
so the API serves the new snapshot.

### `POST /api/admin/reload`

Clears the in-memory caches (`_season_data_cache`, `_similarity_model_cache`,
`_diamond_finder_cache`) without restarting the server.

```bash
curl -X POST http://localhost:8000/api/admin/reload
# => {"status":"ok","cleared":{"season_data":1,"similarity_model":0,"diamond_finder":1}}
```

Call this after any `manage_cache refresh` invocation, or after writing directly
to the SQLite files.

## Workflow Example

```bash
# 1. Scrape contracts
python -m src.scraper --contracts

# 2. Process LEBRON CSV (use the season's raw file)
python -m src.scraper --lebron-csv data/lebron-data-2026.csv --season 2025-26

# 3. Run full refresh with forced NBA API re-fetch
python -m src.manage_cache refresh --season 2025-26 \
    --lebron-csv data/lebron-data-2026.csv --force

# 4. Clear the live API's in-memory caches
curl -X POST http://localhost:8000/api/admin/reload
```

