# Sieve | NBA Contract Analytics

A full-stack analytics platform that identifies market inefficiencies in NBA player contracts by combining salary data with advanced impact metrics.

**[Live Demo](https://sievenba-dashboard.onrender.com)**

---

## Overview

NBA teams spend over $4 billion annually on player salaries, yet traditional statistics poorly predict which contracts provide value. Sieve solves this by:

1. **Computing** a Value Gap score that quantifies how much a player's on-court impact exceeds or falls short of their salary
2. **Visualizing** market inefficiencies through interactive charts and tables
3. **Providing** tools like Diamond Finder (find cheaper replacement players) and Similarity Engine (historical player comparisons)

---

## Key Features

### Value Analysis
- **Value Gap Metric**: Custom algorithm that normalizes salary and impact to a 0-100 scale, then computes a weighted difference
- **Quadrant Chart**: Instantly see which players are underpaid (high impact, low salary) vs overpaid
- **Filterable Tables**: Search and filter by salary range, impact range, team, and player name

### Team Efficiency
- **Payroll vs Wins**: Visualize how efficiently each front office converts spending into wins
- **Team Radar Charts**: Compare two teams across offensive/defensive metrics

### Diamond Finder
- **Archetype-Based Similarity**: Uses scikit-learn KNN with weighted features to find players with similar playstyles
- **Replacement Analysis**: Given any player, find statistically similar players who cost less
- **Style Over Production**: Weights usage rate, assist rate, and archetypes heavily to match role, not just output

### Similarity Engine
- **Historical Comparisons**: Compare any current player to 11,000+ historical player-seasons (2016-present)
- **Multi-Season Support**: See how a player's closest comparisons change year-over-year

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, TypeScript, TailwindCSS, Plotly.js |
| **Backend** | FastAPI, Python 3.11+ |
| **Database** | SQLite (player DB + team DB) |
| **ML** | scikit-learn (KNN, StandardScaler) |
| **Data Sources** | NBA API, BBall Index, Basketball Reference |
| **Deployment** | Render (backend), Vercel-compatible (frontend) |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### Installation

```bash
# Clone the repository
git clone https://github.com/giocld/sieve.git
cd sieve

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running Locally

```bash
# Start both backend and frontend
./run_dev.sh

# Or run separately:
# Terminal 1 - Backend (port 8000)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend (port 5173)
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Updating Data

The database ships with cached data, but you can refresh it with current season stats:

### Quick Refresh (recommended)
```bash
# Refresh all data for current season
python -m src.manage_cache refresh
```

### Full Pipeline
```bash
# 1. Scrape latest contracts from Basketball Reference
python -m src.scraper --contracts

# 2. Download LEBRON CSV from BBall Index (manual step)
#    Save to data/LEBRON.csv

# 3. Process LEBRON and run full refresh
python -m src.manage_cache refresh --lebron-csv data/LEBRON.csv

# 4. (Optional) Check cache status
python -m src.manage_cache status
```

### Data Freshness

| Data Type | Update Frequency | How to Refresh |
|-----------|------------------|----------------|
| LEBRON | Weekly during season | Download CSV from BBall Index, run refresh |
| Contracts | After trades/signings | `python -m src.scraper --contracts` |
| Player Stats | Daily during season | `python -m src.manage_cache refresh` |
| Standings | Auto-fetched when stale | Automatic |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│  Overview | Players | Teams | Similarity                        │
│  (Plotly charts, TanStack Query, TailwindCSS)                   │
└─────────────────────────────────────────────────────────────────┘
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│  /api/players  /api/teams  /api/charts/*  /api/similarity/*     │
│  /api/diamond-finder/*                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ data_processing│    │ visualizations │    │ cache_manager │
│               │    │               │    │               │
│ - Merging     │    │ - Plotly figs │    │ - SQLite ops  │
│ - Value calc  │    │ - Charts      │    │ - Caching     │
│ - KNN models  │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────────┐              ┌───────────────────────┐
│   sieve_players.db    │              │    sieve_teams.db     │
│  - LEBRON metrics     │              │  - Standings          │
│  - Contracts          │              │  - Team efficiency    │
│  - Player analysis    │              │                       │
│  - Similarity model   │              │                       │
└───────────────────────┘              └───────────────────────┘
```

---

## Project Structure

```
sieve/
├── src/
│   ├── api.py              # FastAPI endpoints
│   ├── data_processing.py  # Data loading, merging, ML models
│   ├── visualizations.py   # Plotly chart generators
│   ├── cache_manager.py    # SQLite database operations
│   ├── scraper.py          # Web scraping (contracts, LEBRON)
│   ├── config.py           # Constants and configuration
│   └── manage_cache.py     # CLI for cache management
│
├── frontend/
│   ├── src/
│   │   ├── pages/          # React page components
│   │   ├── components/     # Reusable UI components
│   │   ├── hooks/          # React Query hooks
│   │   └── lib/            # API client, utilities
│   └── package.json
│
├── data/
│   ├── sieve_players.db    # Player database
│   ├── sieve_teams.db      # Team database
│   └── LEBRON.csv          # Raw LEBRON input
│
├── docs/                   # Technical documentation
├── requirements.txt        # Python dependencies
└── run_dev.sh              # Development startup script
```

---

## Documentation

Detailed documentation is available in the `docs/` folder:

- **[Architecture](docs/architecture.md)** - System design and data flow
- **[Algorithms](docs/algorithms.md)** - Value Gap, Diamond Finder, Similarity Engine
- **[Metrics](docs/metrics.md)** - LEBRON, Value Gap, and other metrics explained
- **[Database Schema](docs/database-schema.md)** - SQLite table structures
- **[CLI Reference](docs/cli.md)** - Command-line tools
- **[Data Sources](docs/data-sources.md)** - Where the data comes from

---

## Acknowledgments

- **BBall Index** for LEBRON metrics
- **Basketball Reference** for contract data
- **NBA API** for player/team statistics
