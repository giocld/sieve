# Module Reference

Function-by-function documentation for each Python module.

---

## dashboard.py

Main Dash application with callbacks.

### Global Variables

| Variable | Type | Description |
|----------|------|-------------|
| `app` | Dash | Main Dash application instance |
| `server` | Flask | Flask server for deployment |
| `_season_data_cache` | dict | Cache for loaded season data |
| `_diamond_finder_model` | dict | Cache for Diamond Finder models |
| `df` | DataFrame | Default season player data |
| `df_teams` | DataFrame | Default season team data |
| `df_history` | DataFrame | Historical player stats |
| `knn_model` | NearestNeighbors | Historical similarity model |

### Functions

#### `load_season_data(season)`
Loads all data for a specific NBA season.

**Parameters:**
- `season` (str): Season string, e.g., "2024-25"

**Returns:**
- `tuple`: (df_players, df_teams)

**Process:**
1. Check cache for existing data
2. Call `data_processing.load_and_merge_data()`
3. Call `data_processing.calculate_player_value_metrics()`
4. Call `data_processing.calculate_team_metrics()`
5. Cache and return

---

#### `get_current_data(season=None)`
Wrapper for `load_season_data` with default season.

---

#### `_get_diamond_finder_model(season)`
Gets or builds the Diamond Finder KNN model.

**Parameters:**
- `season` (str): Season string

**Returns:**
- `dict` or `None`: {'model', 'scaler', 'df', 'feature_info'}

**Process:**
1. Check cache for existing model
2. Load and merge data
3. Call `data_processing.build_current_season_similarity()`
4. Cache and return

---

### Callbacks

#### `handle_card_navigation`
Handles clicks on landing page quick-access cards.

**Inputs:** Card click counts (player, team, lineup, similarity)
**Output:** Navigation target ID

---

#### `display_content`
Main navigation controller.

**Inputs:**
- Nav item clicks (home, player, team, lineup, similarity)
- Navigation request from cards
- Selected season

**Outputs:**
- Page content
- View selector state
- Nav item styles (5x)
- Season data store

---

#### `enforce_salary_constraints`
Ensures min salary <= max salary.

---

#### `update_dashboard`
Updates all Player Analysis visualizations.

**Inputs:**
- min_lebron slider
- min_salary slider
- max_salary slider
- search query
- season data store

**Outputs:**
- chart-salary-impact (figure)
- chart-underpaid (figure)
- chart-beeswarm (figure)
- chart-overpaid (figure)
- table-underpaid (html)
- table-overpaid (html)
- table-all-players (html)

---

#### `populate_diamond_finder_dropdown`
Populates player dropdown for Diamond Finder.

**Inputs:** season data store
**Outputs:** dropdown options list

---

#### `update_diamond_finder`
Finds and displays replacement players.

**Inputs:**
- Selected player name
- Season data store

**Outputs:**
- Results container (html)
- Target info card (html)

---

#### `update_team_radar`
Updates team comparison radar chart.

**Inputs:** Two team abbreviations
**Output:** Radar chart figure

---

#### `update_lineup_analysis`
Updates lineup chemistry visualizations.

**Inputs:**
- Team dropdown
- Lineup size radio (2 or 3)
- Min minutes slider

**Outputs:**
- chart-best-lineups (figure)
- chart-worst-lineups (figure)
- chart-lineup-scatter (figure)
- table-best-lineups (html)
- table-worst-lineups (html)

---

#### `update_season_dropdown`
Populates season dropdown for similarity tab.

**Inputs:** Selected player name
**Outputs:** Season options, default value

---

#### `update_similarity_results`
Finds and displays historically similar players.

**Inputs:**
- Player name
- Season
- Exclude self checkbox

**Output:** Results container with player cards

---

## data_processing.py

Data loading, merging, and calculations.

### Functions

#### `load_and_merge_data(lebron_file, season, from_db)`
Loads and merges LEBRON + contract data.

**Parameters:**
- `lebron_file` (str): Path to LEBRON CSV (fallback)
- `season` (str): Season string
- `from_db` (bool): Whether to load from database first

**Returns:**
- `DataFrame` or `tuple`: Merged player data

**Process:**
1. Load LEBRON from DB (or CSV)
2. Load contracts from DB
3. Fuzzy match player names
4. Fetch PLAYER_ID from NBA API
5. Return merged DataFrame

---

#### `calculate_player_value_metrics(df, season)`
Calculates value_gap and related metrics.

**Parameters:**
- `df` (DataFrame): Merged player data
- `season` (str): Season string

**Returns:**
- `DataFrame`: With added columns (salary_norm, impact_norm, value_gap)

**Calculation:**
```python
salary_norm = 100 * (salary - min) / (max - min)
impact_norm = 100 * (LEBRON - min) / (max - min)
value_gap = impact_norm * 1.4 - salary_norm * 0.9 - 10
```

---

#### `calculate_team_metrics(df_players, season)`
Aggregates player data to team level.

**Parameters:**
- `df_players` (DataFrame): Player data with value metrics
- `season` (str): Season string

**Returns:**
- `DataFrame`: Team-level metrics

**Metrics calculated:**
- total_salary: Sum of player salaries
- avg_lebron: Mean LEBRON
- total_war: Sum of LEBRON WAR
- wins, losses: From standings
- efficiency_score: Derived metric

---

#### `build_current_season_similarity(df, season)`
Builds KNN model for Diamond Finder.

**Parameters:**
- `df` (DataFrame): Merged player data
- `season` (str): For fetching NBA API stats

**Returns:**
- `tuple`: (model, scaler, df_filtered, feature_info)

**Features (16):**
USG_PCT, AST_PCT, TS_PCT, REB_PCT, OREB_PCT, DEF_RATING,
archetype_ball_handler, archetype_scorer, archetype_shooter, archetype_big,
defense_rim, defense_perimeter, defense_wing, Minutes, Age, LEBRON

---

#### `find_replacement_players(player_name, df, model, scaler, feature_info, max_results)`
Finds cheaper similar players.

**Parameters:**
- `player_name` (str): Target player
- `df` (DataFrame): From model building
- `model`: KNN model
- `scaler`: StandardScaler
- `feature_info` (dict): Feature config
- `max_results` (int): Number to return

**Returns:**
- `list`: Replacement player dicts

---

#### `build_similarity_model(df_history)`
Builds KNN model for historical similarity.

**Parameters:**
- `df_history` (DataFrame): Historical player stats

**Returns:**
- `tuple`: (model, scaler, df_filtered, feature_info)

---

#### `find_similar_players(player_name, season, df_history, model, scaler, feature_info, exclude_self)`
Finds historically similar players.

**Returns:**
- `list`: Similar player dicts with MatchScore

---

#### `fetch_historical_data()`
Fetches/loads historical player stats (2016-present).

**Returns:**
- `DataFrame`: ~11,000 player-seasons

---

#### `fetch_standings(force_refresh, season)`
Fetches NBA standings from API.

---

#### `get_best_lineups(team_abbr, group_quantity, min_minutes, top_n)`
Gets best performing lineups.

---

#### `get_worst_lineups(...)`
Gets worst performing lineups.

---

#### `get_team_radar_data(team_abbr)`
Gets radar chart data for a team.

---

## cache_manager.py

SQLite database operations.

### Class: CacheManager

Singleton that manages both databases.

#### `__init__()`
Initializes database connections.

**Databases:**
- `sieve_players.db`: Player data
- `sieve_teams.db`: Team data

---

#### Player Database Methods

| Method | Description |
|--------|-------------|
| `save_lebron_metrics(df, season)` | Saves LEBRON data |
| `load_lebron_metrics(season)` | Loads LEBRON data |
| `save_contracts(df, season)` | Saves contract data |
| `load_contracts(season)` | Loads contract data |
| `save_player_analysis(df, season)` | Saves calculated metrics |
| `load_player_analysis(season)` | Loads calculated metrics |
| `save_player_stats(df)` | Saves historical stats |
| `load_player_stats()` | Loads historical stats |

---

#### Team Database Methods

| Method | Description |
|--------|-------------|
| `save_standings(df, season)` | Saves standings |
| `load_standings(season)` | Loads standings |
| `save_team_efficiency(df, season)` | Saves team metrics |
| `load_team_efficiency(season)` | Loads team metrics |
| `save_lineups(df)` | Saves lineup data |
| `load_lineups(team, size, min_minutes)` | Loads lineup data |

---

### Global Instance

```python
cache = CacheManager()  # Singleton used by all modules
```

---

## visualizations.py

Plotly chart generators.

### Functions

#### `create_salary_impact_scatter(df)`
Creates the main salary vs impact scatter plot.

**Encoding:**
- X-axis: LEBRON
- Y-axis: current_year_salary
- Size: LEBRON WAR
- Color: value_gap (green=underpaid, red=overpaid)

---

#### `create_underpaid_bar(df)`
Horizontal bar chart of top 20 underpaid players.

---

#### `create_overpaid_bar(df)`
Horizontal bar chart of top 20 overpaid players.

---

#### `create_player_beeswarm(df)`
Beeswarm plot with player headshots.

---

#### `create_efficiency_quadrant(df_teams)`
Team scatter plot (salary vs wins).

---

#### `create_team_grid(df_teams)`
Team card grid with logos and stats.

---

#### `create_team_radar_chart(data1, data2, team1, team2)`
Radar chart comparing two teams.

---

#### `create_lineup_bar_chart(df, title, color, metric)`
Bar chart for lineup performance.

---

#### `create_lineup_scatter(df_best, df_worst)`
Scatter plot of lineups (OFF vs DEF rating).

---

#### `create_replacement_card(replacement, target_name, target_salary)`
Diamond Finder replacement card.

---

#### `create_diamond_finder_results(replacements, target_name, target_salary, target_lebron)`
Container for all replacement cards.

---

#### `create_player_table(df, table_type)`
DataTable for underpaid/overpaid lists.

---

#### `create_all_players_table(df)`
Full searchable player DataTable.

---

#### `create_lineup_table(df, table_type)`
DataTable for lineup lists.

---

## layout.py

HTML/component structure.

### Functions

#### `create_main_layout()`
Creates the main app layout with navbar and content container.

---

#### `create_landing_tab(df, df_teams)`
Home page with summary stats and quick-access cards.

---

#### `create_player_tab(df)`
Player Analysis tab with filters and chart placeholders.

---

#### `create_team_tab(df_teams, fig_quadrant, fig_grid)`
Team Analysis tab with quadrant chart and grid.

---

#### `create_lineup_tab(team_options)`
Lineup Chemistry tab with team selector.

---

#### `create_similarity_tab(player_options)`
Historical Comps tab with player/season dropdowns.

---

#### `create_similarity_card(name, season, pid, stats, position, match_score, distance, is_target)`
Card for historical similarity results.

---

## scraper.py

Web scraping utilities.

### Functions

#### `scrape_bball_ref()`
Scrapes contract data from Basketball Reference.

**Process:**
1. Request HTML from BBRef
2. Parse with BeautifulSoup
3. Extract salary table
4. Clean and normalize data
5. Save to CSV and database

---

#### `scrape_lebron_data(season)`
Scrapes LEBRON data from BBall Index.

**Process:**
1. Launch Selenium browser
2. Navigate to BBall Index
3. Select season, click "Show All"
4. Download CSV
5. Process and save

---

#### `parse_lebron_csv(filepath, season)`
Processes manually downloaded LEBRON CSV.

---

## config.py

Constants and configuration.

### Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `DATA_DIR` | "data" | Data directory path |
| `PLAYERS_DB` | "data/sieve_players.db" | Player database path |
| `TEAMS_DB` | "data/sieve_teams.db" | Team database path |
| `CURRENT_SEASON` | "2025-26" | Active season |

### Functions

#### `get_previous_season(season)`
Returns the previous season string.

#### `get_season_years(season)`
Parses season into (start_year, end_year).

