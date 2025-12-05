# Data Sources

## LEBRON Data

### Source
**BBall Index:** https://www.bball-index.com/lebron-application/

### What is LEBRON?
LEBRON stands for **L**uck-adjusted player **E**stimate using a **B**ox prior **O**n/off **R**ating **N**ormalized.

It's an all-in-one impact metric that measures a player's contribution per 100 possessions. Unlike box score stats, LEBRON accounts for:
- On/off court impact (how team performs with/without player)
- Luck adjustment (removes variance from 3PT shooting, etc.)
- Box prior (uses box score stats to stabilize small samples)

### How We Get It
1. **Manual download** from BBall Index (export CSV)
2. **Automated scraping** via `scraper.py --lebron` (Selenium)
3. Stored as `data/LEBRON.csv` or `data/LEBRON_2025_26.csv`
4. Cached to SQLite: `sieve_players.db` -> `lebron_metrics` table

### Fields

| Field | Type | Description | Example | Range |
|-------|------|-------------|---------|-------|
| Player | string | Player name | "Shai Gilgeous-Alexander" | - |
| Age | int | Player age | 26 | 19-45 |
| Team(s) | string | Team abbreviation | "OKC" | - |
| Minutes | int | Total minutes played | 2145 | 0-3000 |
| LEBRON | float | Overall impact per 100 poss | 8.67 | -5 to +10 |
| O-LEBRON | float | Offensive component | 6.21 | -5 to +8 |
| D-LEBRON | float | Defensive component | 2.46 | -3 to +5 |
| LEBRON WAR | float | Wins Above Replacement | 15.2 | -2 to +20 |
| Offensive Archetype | string | Offensive style | "Shot Creator" | See below |
| Defensive Role | string | Defensive assignment | "Helper" | See below |
| Rotation Role | string | Playing time tier | "Starter" | See below |

### LEBRON Scale Interpretation

| LEBRON Score | Meaning | Example Players |
|--------------|---------|-----------------|
| +6 to +10 | MVP level | SGA, Jokic, Luka |
| +3 to +6 | All-Star | Cade, Brunson, Tatum |
| +1 to +3 | Quality starter | Randle, Bridges, Reaves |
| -1 to +1 | Average/rotation | Most role players |
| -3 to -1 | Below average | Limited rotation |
| -5 to -3 | Negative impact | End of bench |

### Offensive Archetypes

| Archetype | Description | Typical Usage | Example Players |
|-----------|-------------|---------------|-----------------|
| Shot Creator | Creates own shots off dribble | 25-35% USG | SGA, Luka, Mitchell, Tatum |
| Primary Ball Handler | Main playmaker/initiator | 22-30% USG, high AST | Harden, Brunson, Garland |
| Secondary Ball Handler | Secondary playmaker | 18-25% USG | Reaves, White, Maxey |
| Movement Shooter | Off-ball, relocates constantly | Catch-and-shoot | Curry, Klay, Duncan Robinson |
| Off Screen Shooter | Catches and shoots off screens | Spot-up specialist | Buddy Hield, Korkmaz |
| Stretch Big | Big man who shoots 3s | Spacing big | KAT, Turner, Porzingis |
| Post Scorer | Traditional low-post scorer | Back-to-basket | Jokic, Embiid, Valanciunas |
| Athletic Finisher | Rim runner, lob threat | High FG%, low 3PA | Giannis, AD, Ayton |
| Versatile Big | Big who does multiple things | Varied | Mobley, Holmgren |
| Slasher | Attacks rim off dribble | High paint touches | Ant Edwards, Ja Morant |

### Defensive Roles

| Role | Description | Assignment | Example Players |
|------|-------------|------------|-----------------|
| Anchor Big | Primary rim protector | Protects paint, contests at rim | Gobert, Wemby, Turner |
| Mobile Big | Switchable big man | Can guard perimeter | Bam, Mobley, AD |
| Wing Stopper | Elite wing defender | Guards best wing | OG, Herb Jones, Bridges |
| Point of Attack | Guards the ball handler | Pressures PG | Jrue, Smart, Caruso |
| Chaser | Hounds shooters over screens | Fights through screens | Maxey, Reaves, White |
| Helper | Rotational defender | Help defense, closeouts | Most wings |
| Low Activity | Limited defensive role | Hides on defense | Older stars, offensive specialists |

### Rotation Roles

| Role | Minutes/Game | Description |
|------|--------------|-------------|
| Starter | 30+ | Starts and plays heavy minutes |
| Rotation | 15-30 | Regular rotation player |
| Fringe Rotation | 8-15 | Situational/matchup dependent |
| Deep Bench | <8 | Garbage time only |

---

## Contract Data

### Source
**Basketball Reference:** https://www.basketball-reference.com/contracts/players.html

### How We Get It
1. Scraped via `scraper.py --contracts` (BeautifulSoup)
2. Raw HTML parsed for salary tables
3. Stored as `data/bbref_contracts_raw.csv`
4. Cached to SQLite: `sieve_players.db` -> `contracts` table

### Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| player_name | string | Player name (cleaned) | "Shai Gilgeous-Alexander" |
| bbref_id | string | BBRef player ID | "gilMDgesh01" |
| current_year_salary | int | Salary this season (USD) | 38,292,682 |
| total_contract_value | int | Total guaranteed remaining | 172,317,069 |
| contract_length | int | Years on contract | 5 |
| average_annual_value | int | Average per year | 34,463,414 |
| years_remaining | int | Years left | 4 |
| year_0, year_1, ... | int | Salary each future year | 40,064,220 |

### Salary Scale (2024-25)

| Tier | Salary Range | Description | Examples |
|------|--------------|-------------|----------|
| Supermax | $50M+ | Veteran max extensions | Curry, Giannis, Jokic |
| Max | $40-50M | Standard max contracts | SGA, Luka, Tatum |
| Near-max | $30-40M | Large non-max deals | Brunson, Mitchell |
| Mid-tier | $15-30M | Solid starters | Bridges, Randle |
| Role player | $5-15M | Rotation players | Most bench players |
| Minimum | <$5M | Vet min, two-way | End of roster |

---

## NBA API Data

### Library
`nba_api` Python package (unofficial NBA stats API wrapper)

### Base URL
`https://stats.nba.com/stats/`

### Endpoints Used

#### 1. LeagueDashPlayerStats (Advanced)

**Purpose:** Advanced player metrics for Diamond Finder archetype model

**Endpoint:** `leaguedashplayerstats`

**Parameters:**
```python
{
    'season': '2024-25',
    'per_mode_detailed': 'PerGame',
    'measure_type_detailed_defense': 'Advanced'
}
```

**Fields Used:**

| Field | Description | Formula | Range |
|-------|-------------|---------|-------|
| USG_PCT | Usage Percentage | 100 * ((FGA + 0.44*FTA + TOV) * TmMP) / (MP * (TmFGA + 0.44*TmFTA + TmTOV)) | 10-40% |
| AST_PCT | Assist Percentage | 100 * AST / (((MP/(TmMP/5)) * TmFGM) - FGM) | 5-50% |
| TS_PCT | True Shooting | PTS / (2 * (FGA + 0.44*FTA)) | 40-70% |
| REB_PCT | Rebound Percentage | Total rebounds / available rebounds | 5-25% |
| OREB_PCT | Off Reb Percentage | Offensive rebounds / off reb opportunities | 1-15% |
| DREB_PCT | Def Reb Percentage | Defensive rebounds / def reb opportunities | 10-35% |
| DEF_RATING | Defensive Rating | Points allowed per 100 possessions | 100-125 |
| OFF_RATING | Offensive Rating | Points scored per 100 possessions | 100-130 |
| NET_RATING | Net Rating | OFF_RATING - DEF_RATING | -15 to +20 |

#### 2. LeagueDashPlayerStats (Base)

**Purpose:** Historical player stats for Similarity Engine

**Endpoint:** `leaguedashplayerstats`

**Parameters:**
```python
{
    'season': '2023-24',  # Each season separately
    'per_mode_detailed': 'PerGame',
    'measure_type_detailed_defense': 'Base'
}
```

**Fields Used:**

| Field | Description |
|-------|-------------|
| PLAYER_ID | NBA.com player ID (for headshots) |
| PLAYER_NAME | Player name |
| TEAM_ABBREVIATION | Team |
| GP | Games played |
| MIN | Minutes per game |
| PTS | Points per game |
| REB | Rebounds per game |
| AST | Assists per game |
| STL | Steals per game |
| BLK | Blocks per game |
| TOV | Turnovers per game |
| FG_PCT | Field goal percentage |
| FG3_PCT | 3-point percentage |
| FT_PCT | Free throw percentage |

#### 3. LeagueDashLineups

**Purpose:** Lineup chemistry analysis

**Endpoint:** `leaguedashlineups`

**Parameters:**
```python
{
    'season': '2024-25',
    'group_quantity': 2,  # or 3 for trios
    'per_mode_detailed': 'Totals'
}
```

**Fields Used:**

| Field | Description |
|-------|-------------|
| GROUP_NAME | Player names (e.g., "Player1 - Player2") |
| TEAM_ABBREVIATION | Team |
| GP | Games played together |
| MIN | Minutes played together |
| PLUS_MINUS | Net points while on floor |
| NET_RATING | Point differential per 100 poss |
| OFF_RATING | Points scored per 100 poss |
| DEF_RATING | Points allowed per 100 poss |

#### 4. LeagueStandings

**Purpose:** Team standings and records

**Endpoint:** `leaguestandings`

**Fields Used:**

| Field | Description |
|-------|-------------|
| TeamID | NBA team ID |
| TeamCity | City name |
| TeamName | Team name |
| TeamAbbreviation | 3-letter code |
| Conference | East/West |
| PlayoffRank | Seed (1-15) |
| WINS | Win count |
| LOSSES | Loss count |
| WinPCT | Win percentage |

---

## Data Freshness

| Data Type | Update Frequency | How to Refresh |
|-----------|------------------|----------------|
| LEBRON | Weekly (manual) | Download CSV from BBall Index |
| Contracts | Yearly (or after trades) | `python -m src.scraper --contracts` |
| Player Stats | Daily during season | `python -m src.manage_cache refresh` |
| Standings | Daily during season | Auto-fetched when stale |
| Lineups | On-demand | Fetched per request, cached 24h |

