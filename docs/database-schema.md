# Database Schema

Sieve uses two SQLite databases to separate player and team data.

## Database Files

| Database | Path | Purpose |
|----------|------|---------|
| sieve_players.db | `data/sieve_players.db` | Player-level data |
| sieve_teams.db | `data/sieve_teams.db` | Team-level data |

---

## sieve_players.db

### Table: lebron_metrics

Stores LEBRON impact data from BBall Index.

```sql
CREATE TABLE lebron_metrics (
    player_name TEXT,           -- "Shai Gilgeous-Alexander"
    age INTEGER,                -- 26
    team TEXT,                  -- "OKC"
    minutes INTEGER,            -- 2145
    lebron REAL,                -- 8.67
    o_lebron REAL,              -- 6.21
    d_lebron REAL,              -- 2.46
    lebron_war REAL,            -- 15.2
    offensive_archetype TEXT,   -- "Shot Creator"
    defensive_role TEXT,        -- "Helper"
    rotation_role TEXT,         -- "Starter"
    _season TEXT,               -- "2024-25"
    _updated_at TIMESTAMP       -- "2024-12-04 10:30:00"
);

-- Index for season queries
CREATE INDEX idx_lebron_season ON lebron_metrics(_season);
```

**Row count:** ~300 per season (players with 200+ minutes)

---

### Table: contracts

Stores contract/salary data from Basketball Reference.

```sql
CREATE TABLE contracts (
    player_name TEXT,              -- "Shai Gilgeous-Alexander"
    bbref_id TEXT,                 -- "gilgesh01"
    current_year_salary INTEGER,   -- 38292682
    total_contract_value INTEGER,  -- 172317069
    contract_length INTEGER,       -- 5
    average_annual_value INTEGER,  -- 34463414
    years_remaining INTEGER,       -- 4
    year_0 INTEGER,                -- 38292682 (current year)
    year_1 INTEGER,                -- 40064220 (next year)
    year_2 INTEGER,                -- 41835758
    year_3 INTEGER,                -- 43607296
    year_4 INTEGER,                -- 45378834
    _season TEXT,                  -- "2024-25"
    _updated_at TIMESTAMP
);

CREATE INDEX idx_contracts_season ON contracts(_season);
```

**Row count:** ~500 per season (all rostered players)

---

### Table: player_analysis

Stores calculated metrics (value gap, etc.) for the merged player data.

```sql
CREATE TABLE player_analysis (
    player_name TEXT,              -- "Shai Gilgeous-Alexander"
    PLAYER_ID INTEGER,             -- 1628983 (NBA.com ID for headshots)
    team TEXT,                     -- "OKC"
    age INTEGER,                   -- 26
    minutes INTEGER,               -- 2145
    lebron REAL,                   -- 8.67
    o_lebron REAL,                 -- 6.21
    d_lebron REAL,                 -- 2.46
    lebron_war REAL,               -- 15.2
    current_year_salary INTEGER,   -- 38292682
    total_contract_value INTEGER,  -- 172317069
    salary_norm REAL,              -- 75.3 (0-100 percentile)
    impact_norm REAL,              -- 100.0 (0-100 percentile)
    value_gap REAL,                -- 72.5 (calculated)
    offensive_archetype TEXT,      -- "Shot Creator"
    defensive_role TEXT,           -- "Helper"
    archetype TEXT,                -- "Shot Creator / Helper"
    _season TEXT,                  -- "2024-25"
    _match_score REAL,             -- 95.2 (fuzzy match confidence)
    _updated_at TIMESTAMP
);

CREATE INDEX idx_analysis_season ON player_analysis(_season);
CREATE INDEX idx_analysis_player ON player_analysis(player_name);
```

**Row count:** ~260 per season (players with both LEBRON and contract data)

---

### Table: player_stats

Stores historical player stats for the Similarity Engine.

```sql
CREATE TABLE player_stats (
    PLAYER_ID INTEGER,             -- 1628983
    PLAYER_NAME TEXT,              -- "Shai Gilgeous-Alexander"
    SEASON_ID TEXT,                -- "2023-24"
    TEAM_ABBREVIATION TEXT,        -- "OKC"
    GP INTEGER,                    -- 75
    MIN REAL,                      -- 34.2
    PTS REAL,                      -- 30.1
    REB REAL,                      -- 5.5
    AST REAL,                      -- 6.2
    STL REAL,                      -- 2.0
    BLK REAL,                      -- 0.9
    TOV REAL,                      -- 2.5
    FG_PCT REAL,                   -- 0.535
    FG3_PCT REAL,                  -- 0.353
    FT_PCT REAL,                   -- 0.874
    USG_PCT REAL,                  -- 0.318
    rTS REAL,                      -- 1.08 (relative true shooting)
    AST_PCT REAL,                  -- 0.287
    AST_RATIO REAL,                -- 0.252
    TOV_PCT REAL,                  -- 0.102
    DREB_PCT REAL,                 -- 0.147
    OREB_PCT REAL,                 -- 0.018
    DEF_RATING REAL,               -- 110.2
    OFF_RATING REAL,               -- 124.5
    NET_RATING REAL,               -- 14.3
    POSITION_GROUP TEXT,           -- "guard"
    _updated_at TIMESTAMP
);

CREATE INDEX idx_stats_player ON player_stats(PLAYER_NAME);
CREATE INDEX idx_stats_season ON player_stats(SEASON_ID);
```

**Row count:** ~11,000 (multiple seasons, 2016-present)

---

## sieve_teams.db

### Table: standings

Stores NBA standings from NBA API.

```sql
CREATE TABLE standings (
    TeamID INTEGER,                -- 1610612760
    TeamCity TEXT,                 -- "Oklahoma City"
    TeamName TEXT,                 -- "Thunder"
    TeamAbbreviation TEXT,         -- "OKC"
    Conference TEXT,               -- "West"
    ConferenceRecord TEXT,         -- "25-8"
    PlayoffRank INTEGER,           -- 1
    Division TEXT,                 -- "Northwest"
    DivisionRank INTEGER,          -- 1
    WINS INTEGER,                  -- 35
    LOSSES INTEGER,                -- 12
    WinPCT REAL,                   -- 0.745
    HOME TEXT,                     -- "20-4"
    ROAD TEXT,                     -- "15-8"
    L10 TEXT,                      -- "8-2"
    CurrentStreak TEXT,            -- "W4"
    _season TEXT,                  -- "2024-25"
    _updated_at TIMESTAMP
);

CREATE INDEX idx_standings_season ON standings(_season);
```

**Row count:** 30 per season

---

### Table: team_efficiency

Stores calculated team-level metrics.

```sql
CREATE TABLE team_efficiency (
    team TEXT,                     -- "OKC"
    team_name TEXT,                -- "Thunder"
    total_salary INTEGER,          -- 165000000
    avg_lebron REAL,               -- 2.85
    total_war REAL,                -- 45.2
    wins INTEGER,                  -- 35
    losses INTEGER,                -- 12
    win_pct REAL,                  -- 0.745
    war_per_dollar REAL,           -- 0.000274
    efficiency_score REAL,         -- 85.3
    _season TEXT,                  -- "2024-25"
    _updated_at TIMESTAMP
);

CREATE INDEX idx_efficiency_season ON team_efficiency(_season);
```

**Row count:** 30 per season

---

### Table: lineups

Stores lineup performance data from NBA API.

```sql
CREATE TABLE lineups (
    GROUP_NAME TEXT,               -- "SGA - Dort - Williams"
    TEAM_ABBREVIATION TEXT,        -- "OKC"
    GROUP_QUANTITY INTEGER,        -- 3 (trio)
    GP INTEGER,                    -- 25
    MIN REAL,                      -- 156.4
    PLUS_MINUS REAL,               -- 89.0
    NET_RATING REAL,               -- 18.5
    OFF_RATING REAL,               -- 122.3
    DEF_RATING REAL,               -- 103.8
    _season TEXT,                  -- "2024-25"
    _updated_at TIMESTAMP
);

CREATE INDEX idx_lineups_team ON lineups(TEAM_ABBREVIATION);
CREATE INDEX idx_lineups_size ON lineups(GROUP_QUANTITY);
```

**Row count:** ~5,000 per season (varies by MIN filter)

---

## Multi-Season Support

All tables include a `_season` column (format: "YYYY-YY").

**Saving data:**
```python
# Old season data is deleted before inserting new
DELETE FROM table WHERE _season = '2024-25';
INSERT INTO table (..., _season) VALUES (..., '2024-25');
```

**Loading data:**
```python
# Load specific season
SELECT * FROM table WHERE _season = '2024-25';

# Load all seasons (for historical comparison)
SELECT * FROM table;
```

---

## Schema Migrations

When new columns are added, the cache manager detects schema mismatches:

```python
# In cache_manager.py
existing_cols = set(col[1] for col in cursor.execute("PRAGMA table_info(table)"))
new_cols = set(df.columns)

if new_cols != existing_cols:
    # Drop and recreate table with new schema
    cursor.execute("DROP TABLE table")
    df.to_sql("table", conn)
```

This ensures backward compatibility when the codebase evolves.

