# Sieve Data Sources - Complete Reference

This document provides a comprehensive overview of all data sources used in Sieve, where they're used, how they're processed, and potential improvements.

---

## Table of Contents

1. [Data Architecture Overview](#data-architecture-overview)
2. [Source Data Files](#source-data-files)
3. [Cached Data (SQLite)](#cached-data-sqlite)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Column Reference](#column-reference)
6. [Season Handling](#season-handling)
7. [Known Issues & Solutions](#known-issues--solutions)
8. [Automation Opportunities](#automation-opportunities)

---

## Data Architecture Overview

```
External Sources                    Local Storage                    Application
-----------------                   -------------                    -----------

BBall Reference  ----scraper.py---> basketball_reference_contracts.csv
                                              |
                                              v
ESPN/BBIndex    ----(manual)------> LEBRON.csv ----+
                                                   |
                                                   +---> load_and_merge_data()
                                                   |           |
                                                   |           v
NBA Stats API   ----data_processing.py------------>    sieve_cache.db
  - Lineups                                                    |
  - Team Stats                                                 |
  - Standings                                                  v
  - Historical                                           Dashboard
```

---

## Source Data Files

These files contain raw data that must be manually updated or scraped.

### 1. LEBRON.csv

| Attribute | Value |
|-----------|-------|
| **Source** | ESPN/BBIndex LEBRON metric (manual download) |
| **Location** | `data/LEBRON.csv` |
| **Update Frequency** | Manual - needs refresh when new season data available |
| **Records** | ~560 players (current season) |

#### Columns

| Column | Type | Description | Used In |
|--------|------|-------------|---------|
| `Season` | string | Season ID (e.g., "2024-25") | Filtering |
| `Player` | string | Player full name | Merge key |
| `Age` | int | Player age | Player tab charts |
| `Team(s)` | string | Team abbreviation(s) | Team grouping |
| `Minutes` | int | Total minutes played | Filtering |
| `Rotation Role` | string | Role classification (Star, Starter, Rotation) | Display |
| `Offensive Archetype` | string | Offensive play style | Archetype display |
| `Defensive Role` | string | Defensive play style | Archetype display |
| `LEBRON WAR` | float | Wins Above Replacement | Value calculations |
| `LEBRON` | float | Overall impact metric (-3 to +6 range) | Value Gap calculation |
| `O-LEBRON` | float | Offensive impact | Detailed analysis |
| `D-LEBRON` | float | Defensive impact | Detailed analysis |

#### How to Update

1. Visit [ESPN LEBRON page](https://www.espn.com/nba/stats/player/_/view/lebron) or BBIndex
2. Export current season data
3. Replace `data/LEBRON.csv`
4. Ensure column names match expected format

---

### 2. basketball_reference_contracts.csv

| Attribute | Value |
|-----------|-------|
| **Source** | Basketball Reference (web scrape) |
| **Location** | `data/basketball_reference_contracts.csv` |
| **Update Frequency** | Run scraper when contracts change (offseason, trades) |
| **Records** | ~440 players |

#### Columns

| Column | Type | Description | Used In |
|--------|------|-------------|---------|
| `player_name` | string | Player full name | Merge key |
| `year_4` | float | Guaranteed contract amount | Contract calculations |
| `year_0` | float | 2025-26 salary | Current year salary |
| `year_1` | float | 2026-27 salary | Future projections |
| `year_2` | float | 2027-28 salary | Future projections |
| `contract_length` | int | Total years on contract | Display |
| `total_contract_value` | float | Sum of all years | Display |
| `average_annual_value` | float | AAV | Display |
| `current_year_salary` | float | This season's salary | Value Gap calculation |
| `years_remaining` | int | Years left on deal | Display |

#### How to Update

```bash
cd /home/gio/Workspace/Python/Sieve
source venv/bin/activate
python -m src.scraper
```

**Requirements:**
- Chrome browser installed
- ChromeDriver matching Chrome version
- Internet connection

---

### 3. nba_teams.json

| Attribute | Value |
|-----------|-------|
| **Source** | NBA API (static) |
| **Location** | `data/nba_teams.json` |
| **Update Frequency** | Rarely (only if team changes) |
| **Records** | 30 teams |

#### Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | NBA Team ID |
| `full_name` | string | Full team name |
| `abbreviation` | string | 3-letter code |
| `nickname` | string | Team nickname |
| `city` | string | Team city |
| `state` | string | Team state |
| `year_founded` | int | Franchise founding year |

---

## Cached Data (SQLite)

All API-fetched data is now stored in `data/sieve_cache.db`.

### Database Tables

#### 1. lineups

| Attribute | Value |
|-----------|-------|
| **Source** | NBA API `LeagueDashLineups` |
| **Records** | ~2000 per group size |
| **Cache Duration** | Until manually refreshed |

##### Key Columns

| Column | Type | Description | Used In |
|--------|------|-------------|---------|
| `GROUP_NAME` | string | Player names in lineup | Display |
| `GROUP_ID` | string | Unique lineup identifier | Internal |
| `TEAM_ABBREVIATION` | string | Team code | Filtering |
| `GP` | int | Games played together | Filtering |
| `W` / `L` | int | Wins / Losses | Win % calculation |
| `W_PCT` | float | Win percentage | Display |
| `MIN` | float | Minutes per game | Display |
| `SUM_TIME_PLAYED` | float | Total time (needs /3000 for minutes) | Filtering |
| `PLUS_MINUS` | float | Point differential | Sorting, charts |
| `PTS` | float | Points per game | Display |
| `AST` | float | Assists per game | Display |
| `REB` | float | Rebounds per game | Display |
| `FG_PCT` | float | Field goal percentage | Display |
| `_group_size` | int | 2=duo, 3=trio | Internal filter |
| `_team_filter` | string | Team or "ALL" | Internal filter |
| `_season` | string | Season ID | Internal filter |

---

#### 2. player_stats (Historical)

| Attribute | Value |
|-----------|-------|
| **Source** | NBA API `LeagueDashPlayerStats` + `LeagueDashPtStats` |
| **Records** | ~22,000 (2003-present) |
| **Cache Duration** | Persistent (historical data doesn't change) |

##### Key Columns (155 total)

| Column | Type | Description | Used In |
|--------|------|-------------|---------|
| `PLAYER_ID` | int | NBA Player ID | Headshots, internal |
| `PLAYER_NAME` | string | Player name | Display, matching |
| `TEAM_ABBREVIATION` | string | Team code | Filtering |
| `AGE` | int | Player age in season | Similarity |
| `GP` | int | Games played | Filtering |
| `MIN` | float | Minutes per game | Filtering |
| `PTS` | float | Points (per 100 poss) | Similarity |
| `REB` | float | Rebounds | Similarity |
| `AST` | float | Assists | Similarity |
| `STL` | float | Steals | Similarity |
| `BLK` | float | Blocks | Similarity |
| `TOV` | float | Turnovers | Similarity |
| `USG_PCT` | float | Usage rate (0-100) | Similarity |
| `TS_PCT` | float | True shooting % | Base for rTS |
| `rTS` | float | Relative true shooting | Similarity |
| `AST_PCT` | float | Assist percentage | Similarity |
| `3PA_RATE` | float | 3-point attempt rate | Similarity |
| `FT_PCT` | float | Free throw % | Similarity |
| `FG2_PCT` | float | 2-point FG% | Similarity |
| `DREB_PCT` | float | Defensive rebound % | Similarity |
| `OREB_PCT` | float | Offensive rebound % | Similarity |
| `DEF_RATING` | float | Defensive rating | Similarity |
| `TOUCHES` | float | Touches per game | Ball dominance |
| `AVG_DRIB_PER_TOUCH` | float | Dribbles per touch | Ball dominance |
| `SEASON_ID` | string | Season identifier | Filtering |
| `LEAGUE_AVG_TS` | float | League avg TS% that year | Era adjustment |

---

#### 3. team_stats

| Attribute | Value |
|-----------|-------|
| **Source** | NBA API `LeagueDashTeamStats` (Advanced) |
| **Records** | 30 teams |
| **Cache Duration** | 24 hours recommended |

##### Key Columns

| Column | Type | Description | Used In |
|--------|------|-------------|---------|
| `TEAM_ID` | int | NBA Team ID | Logo URLs |
| `TEAM_NAME` | string | Full team name | Display |
| `OFF_RATING` | float | Offensive rating | Radar chart |
| `DEF_RATING` | float | Defensive rating | Radar chart |
| `NET_RATING` | float | Net rating | Analysis |
| `REB_PCT` | float | Rebounding % | Radar chart |
| `AST_PCT` | float | Assist % | Radar chart |
| `TS_PCT` | float | True shooting % | Radar chart |
| `_stat_type` | string | "advanced" or "base" | Internal |
| `_season` | string | Season ID | Internal |

---

#### 4. standings

| Attribute | Value |
|-----------|-------|
| **Source** | NBA API `LeagueStandings` |
| **Records** | 30 teams |
| **Cache Duration** | 24 hours recommended |

##### Key Columns

| Column | Type | Description | Used In |
|--------|------|-------------|---------|
| `TeamID` | int | NBA Team ID | Matching |
| `TeamCity` | string | City name | Full name construction |
| `TeamName` | string | Nickname | Full name construction |
| `WINS` | int | Season wins | Efficiency Index |
| `LOSSES` | int | Season losses | Display |
| `WinPCT` | float | Win percentage | Display |
| `_season` | string | Season ID | Internal |

---

## Data Flow Diagram

```
+------------------+     +------------------------+     +------------------+
|   LEBRON.csv     |     | basketball_reference_  |     |  NBA Stats API   |
| (Manual Update)  |     |    contracts.csv       |     |                  |
+--------+---------+     | (Scraper Update)       |     +--------+---------+
         |               +------------+-----------+              |
         |                            |                          |
         v                            v                          v
+--------+----------------------------+--------+    +------------+------------+
|           load_and_merge_data()              |    |  fetch_lineup_data()    |
|  - Merges LEBRON + Contracts                 |    |  fetch_historical_data()|
|  - Cleans data types                         |    |  fetch_nba_advanced()   |
|  - Creates archetype strings                 |    |  fetch_standings()      |
+---------------------+------------------------+    +------------+------------+
                      |                                          |
                      v                                          v
         +------------+------------+                +------------+------------+
         | calculate_player_value_ |                |    sieve_cache.db       |
         |       metrics()         |                | (Unified SQLite Cache)  |
         | - Value Gap calculation |                +------------+------------+
         +------------+------------+                             |
                      |                                          |
                      v                                          v
         +------------+------------+                +------------+------------+
         | calculate_team_metrics()|                | Cache Load Functions    |
         | - Aggregates by team    |<---------------| cache.load_lineups()    |
         | - Efficiency Index      |                | cache.load_standings()  |
         +------------+------------+                +-------------------------+
                      |
                      v
         +------------+-------------------------+
         |          Dashboard                   |
         | - Player Tab (charts, tables)       |
         | - Team Tab (quadrant, grid, radar)  |
         | - Lineup Tab (best/worst lineups)   |
         | - Similarity Tab (player comps)     |
         +-------------------------------------+
```

---

## Column Reference

### Merged Player DataFrame (df)

After `load_and_merge_data()` + `calculate_player_value_metrics()`:

| Column | Source | Type | Description |
|--------|--------|------|-------------|
| `player_name` | Both | string | Player name (merge key) |
| `Season` | LEBRON | string | Season ID |
| `Age` | LEBRON | int | Player age |
| `Team(s)` | LEBRON | string | Team abbreviation(s) |
| `Minutes` | LEBRON | int | Minutes played |
| `Rotation Role` | LEBRON | string | Star/Starter/Rotation |
| `Offensive Archetype` | LEBRON | string | Offensive style |
| `Defensive Role` | LEBRON | string | Defensive style |
| `archetype` | Calculated | string | Combined "Off / Def" string |
| `LEBRON WAR` | LEBRON | float | Wins above replacement |
| `LEBRON` | LEBRON | float | Overall impact |
| `O-LEBRON` | LEBRON | float | Offensive impact |
| `D-LEBRON` | LEBRON | float | Defensive impact |
| `current_year_salary` | Contracts | float | This year's salary |
| `year_0` to `year_4` | Contracts | float | Future salaries |
| `contract_length` | Contracts | int | Contract years |
| `total_contract_value` | Contracts | float | Total contract |
| `salary_norm` | Calculated | float | Normalized salary (0-100) |
| `impact_norm` | Calculated | float | Normalized impact (0-100) |
| `value_gap` | Calculated | float | Impact - Salary differential |
| `PLAYER_ID` | Historical | int | NBA ID for headshots |

### Team DataFrame (df_teams)

After `calculate_team_metrics()` + `add_team_logos()`:

| Column | Source | Type | Description |
|--------|--------|------|-------------|
| `Abbrev` | Derived | string | Team abbreviation |
| `Total_Payroll` | Aggregated | float | Sum of player salaries |
| `Total_WAR` | Aggregated | float | Sum of LEBRON WAR |
| `LEBRON` | Aggregated | float | Average player LEBRON |
| `player_name` | Aggregated | int | Roster count |
| `WINS` | Standings | int | Season wins |
| `LOSSES` | Standings | int | Season losses |
| `Cost_Per_Win` | Calculated | float | Payroll / Wins |
| `Efficiency_Index` | Calculated | float | (2*Z_Wins) - Z_Payroll |
| `Payroll_Display` | Formatted | string | "$XXX.XM" format |
| `CPW_Display` | Formatted | string | "$X.XXM" format |
| `TeamID` | NBA API | int | Official team ID |
| `Logo_URL` | Derived | string | CDN URL for logo |

---

## Season Handling

### Current Hardcoded Seasons

| File | Location | Current Value | Issue |
|------|----------|---------------|-------|
| `cache_manager.py` | Default params | `'2024-25'` | Will be stale |
| `data_processing.py` | `fetch_lineup_data()` | `'2024-25'` | Will be stale |
| `data_processing.py` | `fetch_nba_advanced_stats()` | `'2024-25'` | Will be stale |
| `data_processing.py` | `fetch_standings()` | `'2024-25'` | Will be stale |
| `data_processing.py` | `calculate_team_metrics()` | `'2024-25'` | Will be stale |
| `layout.py` | Footer text | `"2024-25 NBA Season"` | Display only |
| `scraper.py` | Column index | `"2025-26"` | Correct for next season |

### Dynamic Season Detection

The codebase already has a `get_season_list()` function that calculates the current season:

```python
def get_season_list(start_year=2014):
    current_date = datetime.now()
    
    # NBA season starts in October
    if current_date.month >= 10:
        current_season_start = current_date.year
    else:
        current_season_start = current_date.year - 1
    
    # Returns list like ['2014-15', '2015-16', ..., '2024-25']
```

### Recommended Fix

Create a central season configuration:

```python
# src/config.py
from datetime import datetime

def get_current_season():
    """Returns current NBA season string (e.g., '2024-25')."""
    now = datetime.now()
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-{str(year + 1)[-2:]}"

CURRENT_SEASON = get_current_season()
```

Then replace all hardcoded `'2024-25'` with `CURRENT_SEASON`.

---

## Known Issues & Solutions

### Issue 1: Stale Season Data

**Problem:** When the 2025-26 season starts (October 2025), all hardcoded `'2024-25'` values will fetch old data.

**Solution:** Implement `src/config.py` with dynamic season detection (see above).

**Files to Update:**
- `src/cache_manager.py` (6 locations)
- `src/data_processing.py` (4 locations)
- `src/layout.py` (1 location - display text)

---

### Issue 2: Manual LEBRON Data Updates

**Problem:** LEBRON.csv must be manually downloaded and replaced.

**Solution Options:**
1. **Scraper:** Build scraper for ESPN LEBRON page (complex, may break)
2. **Scheduled reminder:** GitHub Action to create issue monthly
3. **Alternative metric:** Calculate WAR from NBA API data directly

---

### Issue 3: Contract Data Staleness

**Problem:** Contracts change during offseason, trades, extensions.

**Solution:** 
- Run scraper at start of season (October)
- Run scraper after trade deadline (February)
- Run scraper after free agency (July)

**Automation:**
```yaml
# .github/workflows/update-contracts.yml
name: Update Contracts
on:
  schedule:
    - cron: '0 6 1 10,2,7 *'  # Oct 1, Feb 1, Jul 1
  workflow_dispatch:  # Manual trigger

jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install selenium beautifulsoup4 pandas
          # Install Chrome
      - name: Run scraper
        run: python -m src.scraper
      - name: Commit changes
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add data/basketball_reference_contracts.csv
          git commit -m "Auto-update contracts" || exit 0
          git push
```

---

### Issue 4: NBA API Rate Limiting

**Problem:** NBA Stats API can block requests or timeout.

**Current Mitigations:**
- Custom browser headers
- Retry logic (3 attempts)
- 60-second timeout
- 1-second delay between calls

**Additional Solutions:**
- Implement exponential backoff
- Cache aggressively (already done with SQLite)
- Add request pooling for batch operations

---

### Issue 5: Player Name Mismatches

**Problem:** Names may differ between sources:
- "Marcus Morris" vs "Marcus Morris Sr."
- "Nic Claxton" vs "Nicolas Claxton"

**Current Handling:** Inner join drops unmatched players.

**Solution Options:**
1. Maintain manual alias mapping
2. Use fuzzy matching (fuzzywuzzy library)
3. Match on player ID instead of name (requires ID in both sources)

---

## Automation Opportunities

### 1. Automated Season Transition

**Goal:** Automatically use correct season when October arrives.

**Implementation:**
```python
# src/config.py
from datetime import datetime

def get_current_season():
    now = datetime.now()
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-{str(year + 1)[-2:]}"

def get_season_for_display():
    """Returns human-readable season (e.g., '2024-25 NBA Season')"""
    return f"{get_current_season()} NBA Season"

CURRENT_SEASON = get_current_season()
```

---

### 2. Automated Contract Updates (GitHub Actions)

See Issue 3 above for workflow YAML.

---

### 3. Cache Refresh Scheduling

**Goal:** Automatically refresh standings/team stats during season.

**Implementation Options:**

**Option A: Cron job on server**
```bash
# crontab -e
0 6 * * * cd /path/to/sieve && python -c "
from src.data_processing import fetch_standings, fetch_nba_advanced_stats
fetch_standings(force_refresh=True)
fetch_nba_advanced_stats(force_refresh=True)
"
```

**Option B: In-app auto-refresh**
```python
# In cache_manager.py
def load_standings(self, season='2024-25', auto_refresh_hours=24):
    if not self.is_cache_fresh('standings', max_age_hours=auto_refresh_hours):
        # Trigger background refresh
        from src.data_processing import fetch_standings
        fetch_standings(force_refresh=True, season=season)
    return self._load_from_db('standings', season)
```

---

### 4. LEBRON Data Alternative

**Goal:** Remove dependency on manual LEBRON updates.

**Option: Calculate similar metric from NBA API**

```python
def calculate_impact_metric(df_player_stats):
    """
    Calculate a LEBRON-like impact metric from NBA API data.
    Uses Box Plus/Minus approximation.
    """
    # Simplified BPM calculation
    df = df_player_stats.copy()
    
    # Offensive component
    df['O_IMPACT'] = (
        df['PTS'] * 0.3 + 
        df['AST'] * 0.4 + 
        df['TS_PCT'] * 20 - 
        df['TOV'] * 0.5
    ) / 10
    
    # Defensive component  
    df['D_IMPACT'] = (
        df['STL'] * 0.5 + 
        df['BLK'] * 0.3 + 
        df['DREB_PCT'] * 0.1
    ) / 10
    
    df['IMPACT'] = df['O_IMPACT'] + df['D_IMPACT']
    
    return df
```

---

### 5. Data Validation Pipeline

**Goal:** Catch data quality issues before they break the dashboard.

```python
# src/validate_data.py
def validate_lebron_data(df):
    """Validate LEBRON.csv data quality."""
    issues = []
    
    # Check required columns
    required = ['Player', 'Team(s)', 'LEBRON', 'LEBRON WAR']
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check for nulls in key columns
    for col in ['Player', 'LEBRON']:
        null_count = df[col].isna().sum()
        if null_count > 0:
            issues.append(f"{col} has {null_count} null values")
    
    # Check season is current
    if 'Season' in df.columns:
        seasons = df['Season'].unique()
        if len(seasons) > 1:
            issues.append(f"Multiple seasons in data: {seasons}")
    
    # Check reasonable value ranges
    if 'LEBRON' in df.columns:
        min_val, max_val = df['LEBRON'].min(), df['LEBRON'].max()
        if min_val < -5 or max_val > 8:
            issues.append(f"LEBRON values out of range: [{min_val}, {max_val}]")
    
    return issues

def validate_all():
    """Run all validation checks."""
    import pandas as pd
    
    results = {}
    
    # Validate LEBRON
    df_lebron = pd.read_csv('data/LEBRON.csv')
    results['LEBRON'] = validate_lebron_data(df_lebron)
    
    # Add more validators...
    
    for source, issues in results.items():
        if issues:
            print(f"[WARN] {source}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"[OK] {source}")
    
    return results
```

---

## Quick Reference: Update Checklist

### Start of New Season (October)

- [ ] Update `LEBRON.csv` with new season data
- [ ] Run contract scraper: `python -m src.scraper`
- [ ] Clear old cache: `python -m src.manage_cache clear`
- [ ] Verify season auto-detection works (if implemented)
- [ ] Update display text in `layout.py` if hardcoded

### Mid-Season Updates

- [ ] Refresh `LEBRON.csv` for updated impact metrics
- [ ] Refresh standings: `fetch_standings(force_refresh=True)`
- [ ] Check for traded player contract updates

### Trade Deadline / Free Agency

- [ ] Run contract scraper: `python -m src.scraper`
- [ ] Refresh `LEBRON.csv` for new team assignments

---

## File Locations Summary

| File | Purpose | Update Frequency |
|------|---------|-----------------|
| `data/LEBRON.csv` | Player impact metrics | Monthly during season |
| `data/basketball_reference_contracts.csv` | Contract data | Offseason + trade deadline |
| `data/nba_teams.json` | Team metadata | Rarely |
| `data/sieve_cache.db` | All cached API data | Automatic |
| `src/cache_manager.py` | Cache operations | Code changes only |
| `src/data_processing.py` | Data transformations | Code changes only |
| `src/scraper.py` | Contract scraping | Code changes only |
| `src/dashboard.py` | Main application | Code changes only |

