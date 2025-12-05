# Architecture

## Overview

Sieve is a Dash web application that analyzes NBA player value by combining impact metrics (LEBRON) with contract data to identify overpaid/underpaid players.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Dash 2.x + Plotly 5.x + Bootstrap 5 |
| Backend | Python 3.11 |
| Database | SQLite 3 (2 separate databases) |
| Data Sources | BBall Index, Basketball Reference, NBA API |
| ML | scikit-learn (KNN, StandardScaler) |

## File Structure

```
sieve/
├── src/
│   ├── app.py              # Entry point, imports dashboard
│   ├── dashboard.py        # Dash app initialization, all callbacks
│   ├── layout.py           # HTML structure, component definitions
│   ├── visualizations.py   # Plotly figure generators
│   ├── data_processing.py  # Data loading, merging, calculations
│   ├── cache_manager.py    # SQLite database operations
│   ├── scraper.py          # Web scraping (contracts, LEBRON)
│   ├── config.py           # Constants, paths, current season
│   └── manage_cache.py     # CLI cache management tool
│
├── data/
│   ├── sieve_players.db    # Player database (LEBRON, contracts, analysis)
│   ├── sieve_teams.db      # Team database (standings, efficiency, lineups)
│   ├── LEBRON.csv          # 2024-25 LEBRON data (raw input)
│   └── LEBRON_2025_26.csv  # 2025-26 LEBRON data (raw input)
│
├── docs/                   # This documentation
├── venv/                   # Python virtual environment
└── requirements.txt        # Python dependencies
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                       │
├─────────────────┬─────────────────────┬─────────────────────────────┤
│   BBall Index   │  Basketball Ref     │         NBA API             │
│   (LEBRON CSV)  │  (Contracts HTML)   │  (Stats/Lineups/Standings)  │
└────────┬────────┴──────────┬──────────┴──────────────┬──────────────┘
         │                   │                         │
         │    scraper.py     │      scraper.py         │   nba_api
         │    (manual DL)    │    (BeautifulSoup)      │   (Python pkg)
         │                   │                         │
         ▼                   ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        cache_manager.py                             │
│  ┌──────────────────────┐    ┌──────────────────────┐               │
│  │   sieve_players.db   │    │    sieve_teams.db    │               │
│  │  - lebron_metrics    │    │  - standings         │               │
│  │  - contracts         │    │  - team_efficiency   │               │
│  │  - player_analysis   │    │  - lineups           │               │
│  │  - player_stats      │    │                      │               │
│  └──────────────────────┘    └──────────────────────┘               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       data_processing.py                            │
│                                                                     │
│  load_and_merge_data()     - Combines LEBRON + contracts            │
│  calculate_player_value()  - Computes value_gap                     │
│  calculate_team_metrics()  - Aggregates to team level               │
│  build_similarity_model()  - Trains KNN for historical comps        │
│  build_current_similarity()- Trains KNN for Diamond Finder          │
│  find_replacement_players()- Queries Diamond Finder model           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         dashboard.py                                │
│                                                                     │
│  Dash App Initialization                                            │
│  Navigation Callbacks (tab switching)                               │
│  Player Analysis Callbacks (filters -> charts)                      │
│  Diamond Finder Callbacks (player select -> replacements)           │
│  Team Analysis Callbacks (team select -> radar)                     │
│  Lineup Callbacks (team/size -> best/worst lineups)                 │
│  Similarity Callbacks (player/season -> historical comps)           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│        layout.py            │ │     visualizations.py       │
│                             │ │                             │
│  create_main_layout()       │ │  create_salary_impact()     │
│  create_player_tab()        │ │  create_underpaid_bar()     │
│  create_team_tab()          │ │  create_beeswarm()          │
│  create_lineup_tab()        │ │  create_team_radar()        │
│  create_similarity_tab()    │ │  create_lineup_chart()      │
│  create_landing_tab()       │ │  create_replacement_card()  │
└─────────────────────────────┘ └─────────────────────────────┘
```

## Request Flow Example

**User filters players by salary:**

```
1. User moves salary slider
      │
      ▼
2. Dash callback triggered: update_dashboard()
      │
      ▼
3. Get data: get_current_data(season)
      │
      ├── Cache hit? Return cached (df_players, df_teams)
      │
      └── Cache miss? 
            │
            ├── load_and_merge_data() 
            │     ├── cache.load_lebron_metrics()
            │     ├── cache.load_contracts()
            │     └── fuzzy_match_players()
            │
            ├── calculate_player_value_metrics()
            │
            └── Cache result
      │
      ▼
4. Filter DataFrame by slider values
      │
      ▼
5. Recalculate value_gap for filtered subset
      │
      ▼
6. Generate visualizations:
      ├── visualizations.create_salary_impact_scatter()
      ├── visualizations.create_underpaid_bar()
      ├── visualizations.create_beeswarm()
      └── visualizations.create_overpaid_bar()
      │
      ▼
7. Return figures to Dash, browser updates
```

## Module Dependencies

```
app.py
  └── dashboard.py
        ├── data_processing.py
        │     └── cache_manager.py
        │           └── config.py
        ├── visualizations.py
        └── layout.py

scraper.py (standalone)
  └── cache_manager.py

manage_cache.py (CLI)
  ├── cache_manager.py
  └── data_processing.py
```

