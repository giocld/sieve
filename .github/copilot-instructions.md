# Copilot Instructions for Sieve NBA Analytics

## Project Architecture
- **src/**: All core logic. Key modules:
  - `app.py`: Main entry point for the dashboard (likely Dash/Plotly or Flask).
  - `dashboard.py`, `layout.py`, `visualizations.py`: UI, layout, and plotting logic.
  - `data_processing.py`: Data cleaning, merging, and transformation routines.
  - `scraper.py`: Scrapes NBA contract data from Basketball Reference using Selenium and BeautifulSoup. Outputs to `data/basketball_reference_contracts.csv`.
- **data/**: All CSVs used as input/output. Never hand-edit these; always regenerate via scripts.
- **assets/**: Custom CSS for dashboard styling.
- **docs/**: In-depth code and statistics documentation.
- **requirements.txt**: All dependencies (Selenium, pandas, bs4, etc.).

## Data Flow
- Scraping (`scraper.py`) → Raw contract CSV → Data processing/merging (`data_processing.py`) → Dashboard display (`app.py` and UI modules).
- LEBRON and other advanced stats are loaded from CSVs in `data/`.

## Developer Workflows
- **Scraping**: Run `python src/scraper.py` to update contract data.
- **Dashboard**: Run `python src/app.py` to launch the dashboard locally.
- **Dependencies**: Install with `pip install -r requirements.txt`.
- **Data Regeneration**: Never edit CSVs by hand—always use scripts.

## Conventions & Patterns
- Use pandas DataFrames for all data manipulation.
- All data cleaning and merging logic lives in `data_processing.py`.
- Scraper is robust to table structure changes (maps columns by header name, not index).
- Output CSVs are always written to `data/`.
- Use `assets/custom.css` for all dashboard styling overrides.

## Integration Points
- **External**: Basketball Reference (contracts), LEBRON stats (CSV), NBA stats (CSV/json).
- **Internal**: All modules import from `src/`.

## Examples
- To add a new data source, place the file in `data/` and update `data_processing.py`.
- To change dashboard visuals, edit `visualizations.py` and/or `layout.py`.

## References
- See `README.md` for high-level overview.
- See `docs/CODE_DOCUMENTATION.md` and `docs/STATISTICS_EXPLAINED.md` for details.
