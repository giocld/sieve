# Algorithms

Detailed explanation of the algorithms used in Sieve.

---

## Value Gap Calculation

### Purpose
Identify overpaid and underpaid NBA players by comparing their on-court impact to their salary.

### Location
`data_processing.py` -> `calculate_player_value_metrics()`

### Algorithm

```
INPUT:
  - df: DataFrame with 'LEBRON' and 'current_year_salary' columns

STEP 1: Normalize salary to 0-100 scale
  salary_min = min(all salaries)
  salary_max = max(all salaries)
  For each player:
    salary_norm = 100 * (salary - salary_min) / (salary_max - salary_min)

STEP 2: Normalize LEBRON to 0-100 scale
  lebron_min = min(all LEBRON scores)
  lebron_max = max(all LEBRON scores)
  For each player:
    impact_norm = 100 * (LEBRON - lebron_min) / (lebron_max - lebron_min)

STEP 3: Calculate value gap
  For each player:
    value_gap = impact_norm * 1.4 - salary_norm * 0.9 - 10

OUTPUT:
  - df with added columns: salary_norm, impact_norm, value_gap
```

### Weight Rationale

| Weight | Value | Reason |
|--------|-------|--------|
| Impact multiplier | 1.4 | Impact matters more than salary |
| Salary multiplier | 0.9 | Slight discount on salary importance |
| Offset | -10 | Centers distribution around 0 |

### Edge Cases

- **Filtered data:** When users filter by salary/LEBRON, percentiles are recalculated for the filtered pool only
- **Missing data:** Players without salary data are excluded
- **Negative LEBRON:** Handled normally (still maps to 0-100 scale)

---

## Diamond Finder (Archetype-Based Similarity)

### Purpose
Find statistically similar players who cost less - for replacement analysis.

### Location
- Model building: `data_processing.py` -> `build_current_season_similarity()`
- Finding replacements: `data_processing.py` -> `find_replacement_players()`

### Key Insight
Traditional similarity models weight production heavily. This model weights **STYLE** heavily and **PRODUCTION** lightly. A backup point guard should match with other backup point guards, even if their overall impact differs.

### Algorithm: Model Building

```
INPUT:
  - df: Merged player data with LEBRON metrics
  - season: For fetching NBA API advanced stats

STEP 1: Fetch advanced stats from NBA API
  Call leaguedashplayerstats endpoint
  Get: USG_PCT, AST_PCT, TS_PCT, REB_PCT, OREB_PCT, DEF_RATING
  
STEP 2: Fuzzy match to merge
  For each player in df:
    Find closest match in API data (90% threshold)
    Merge advanced stats columns
    
STEP 3: Create archetype binary features
  archetype_ball_handler = 1 if "Primary Ball Handler" or "Secondary Ball Handler"
  archetype_scorer = 1 if "Shot Creator"
  archetype_shooter = 1 if "Movement Shooter" or "Off Screen Shooter"
  archetype_big = 1 if "Post Scorer" or "Stretch Big" or "Athletic Finisher"
  defense_rim = 1 if "Anchor Big" or "Mobile Big"
  defense_perimeter = 1 if "Chaser" or "Point of Attack"
  defense_wing = 1 if "Wing Stopper"

STEP 4: Define feature weights
  Feature weights (higher = more important for similarity):
  - USG_PCT: 2.5 (most important - defines offensive role)
  - AST_PCT: 2.5 (playmaking vs scoring)
  - TS_PCT: 2.0 (efficiency style)
  - REB_PCT: 1.8 (glass presence)
  - OREB_PCT: 1.5 (big man indicator)
  - DEF_RATING: 1.5 (defensive impact)
  - archetype_*: 1.5-2.0 (role classification)
  - defense_*: 1.2-1.5 (defensive role)
  - Minutes: 0.3 (low - don't match by playing time)
  - Age: 0.2 (very low - don't match by age)
  - LEBRON: 0.5 (intentionally low - match by style, not goodness)

STEP 5: Standardize features
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(feature_matrix)

STEP 6: Apply weights
  X_weighted = X_scaled * feature_weights

STEP 7: Train KNN
  model = NearestNeighbors(n_neighbors=40, metric='cosine')
  model.fit(X_weighted)

OUTPUT:
  - model: Trained KNN model
  - scaler: Fitted StandardScaler
  - df_filtered: DataFrame with all features
  - feature_info: Dict with feature names and weights
```

### Algorithm: Finding Replacements

```
INPUT:
  - player_name: Target player to replace
  - df: DataFrame from model building
  - model, scaler, feature_info: From model building
  - max_results: Number of replacements to return (default 8)

STEP 1: Get target player's feature vector
  target_row = df[df['player_name'] == player_name]
  target_vector = target_row[features].values

STEP 2: Transform target vector
  target_scaled = scaler.transform(target_vector)
  target_weighted = target_scaled * weights

STEP 3: Query KNN
  distances, indices = model.kneighbors(target_weighted, n_neighbors=60)

STEP 4: Filter candidates
  For each neighbor:
    SKIP if same player
    SKIP if salary >= target_salary * 0.9 (not cheaper)
    SKIP if position not compatible:
      - bigs only match bigs
      - guards match guards/wings/versatile
      - wings match guards/wings/versatile

STEP 5: Score each candidate
  # Base score from KNN distance (0-60 points)
  dist_normalized = (distance - min_dist) / (max_dist - min_dist)
  base_score = 60 * (1 - dist_normalized)
  
  # Archetype bonus (-10 to +20)
  if exact_archetype_match: archetype_score = +20
  elif same_archetype_family: archetype_score = +8
  else: archetype_score = -10
  
  # Defensive role bonus (0 to +5)
  if same_defensive_role: defense_score = +5
  elif same_defense_family: defense_score = +2
  else: defense_score = 0
  
  # Advanced stats bonus (0 to +15)
  style_score = 0
  if abs(USG_diff) < 0.03: style_score += 5
  elif abs(USG_diff) < 0.06: style_score += 3
  elif abs(USG_diff) < 0.10: style_score += 1
  
  if abs(AST_diff) < 0.05: style_score += 5
  elif abs(AST_diff) < 0.10: style_score += 3
  elif abs(AST_diff) < 0.15: style_score += 1
  
  if abs(TS_diff) < 0.03: style_score += 5
  elif abs(TS_diff) < 0.05: style_score += 3
  elif abs(TS_diff) < 0.08: style_score += 1
  
  # Final score
  match_score = base_score + archetype_score + defense_score + style_score
  match_score = clamp(match_score, 30, 98)

STEP 6: Sort and return
  Sort by match_score descending
  Return top max_results

OUTPUT:
  List of replacement dicts with:
  - player_name, PLAYER_ID, salary
  - LEBRON, O-LEBRON, D-LEBRON
  - archetype, defense_role, position_group
  - match_score, distance
  - savings, savings_pct
  - USG_PCT, AST_PCT, TS_PCT (for display)
```

### Archetype Families

```python
ARCHETYPE_FAMILIES = {
    'Shot Creator': ['Primary Ball Handler', 'Secondary Ball Handler'],
    'Primary Ball Handler': ['Shot Creator', 'Secondary Ball Handler'],
    'Secondary Ball Handler': ['Shot Creator', 'Primary Ball Handler'],
    'Movement Shooter': ['Off Screen Shooter', 'Stretch Big'],
    'Off Screen Shooter': ['Movement Shooter'],
    'Stretch Big': ['Movement Shooter', 'Versatile Big'],
    'Post Scorer': ['Athletic Finisher', 'Versatile Big'],
    'Athletic Finisher': ['Post Scorer', 'Slasher'],
    'Versatile Big': ['Stretch Big', 'Post Scorer'],
    'Slasher': ['Athletic Finisher', 'Shot Creator'],
}

DEFENSE_FAMILIES = {
    'Anchor Big': ['Mobile Big'],
    'Mobile Big': ['Anchor Big', 'Helper'],
    'Chaser': ['Point of Attack'],
    'Point of Attack': ['Chaser', 'Wing Stopper'],
    'Wing Stopper': ['Point of Attack', 'Helper'],
    'Helper': ['Wing Stopper', 'Mobile Big'],
    'Low Activity': ['Helper'],
}
```

### Position Compatibility

```python
POSITION_COMPATIBILITY = {
    'big': ['big'],  # Bigs only match bigs
    'guard': ['guard', 'wing', 'versatile'],
    'wing': ['guard', 'wing', 'versatile'],
    'versatile': ['guard', 'wing', 'versatile'],
}
```

---

## Historical Similarity Engine

### Purpose
Find the most statistically similar player-seasons in NBA history (2016-present).

### Location
- Model building: `data_processing.py` -> `build_similarity_model()`
- Finding similar: `data_processing.py` -> `find_similar_players()`

### Difference from Diamond Finder
- Uses historical data (11,000+ player-seasons)
- Uses traditional box score stats
- Weights production more heavily
- Allows comparing current players to historical seasons

### Algorithm: Model Building

```
INPUT:
  - df_history: Historical player stats (2016-present)

STEP 1: Filter qualified players
  df_filtered = df[df['GP'] >= 15]  # At least 15 games

STEP 2: Define features and weights
  FEATURE_CONFIG = {
    # Production (moderate weight)
    'PTS': 1.0, 'REB': 1.0, 'AST': 1.0,
    'STL': 0.8, 'BLK': 0.8, 'TOV': 0.8,
    
    # Efficiency/Style (high weight)
    'USG_PCT': 1.5, 'rTS': 1.5, 'AST_PCT': 1.5,
    '3PA_RATE': 1.5, 'FT_PCT': 1.2, 'FG2_PCT': 1.2,
    
    # Defense (moderate-high)
    'DREB_PCT': 1.3, 'OREB_PCT': 1.0, 'DEF_RATING': 1.2,
    
    # Playmaking
    'TOV_AST_RATIO': 1.0,
  }

STEP 3: Classify position groups
  For each player:
    if BLK > 1.0 and REB > 6: position = 'big'
    elif AST > 5: position = 'guard'
    else: position = 'wing'

STEP 4: Standardize and weight
  X_scaled = StandardScaler().fit_transform(X)
  X_weighted = X_scaled * weights

STEP 5: Train KNN
  model = NearestNeighbors(n_neighbors=30, metric='cosine')
  model.fit(X_weighted)

OUTPUT:
  - model, scaler, df_filtered, feature_info
```

### Algorithm: Finding Similar Players

```
INPUT:
  - player_name: Target player
  - season: Target season (e.g., '2023-24')
  - exclude_self: Whether to exclude other seasons of same player

STEP 1: Find target player-season
  target_row = df[(PLAYER_NAME == player_name) & (SEASON_ID == season)]

STEP 2: Transform target
  target_scaled = scaler.transform(target_row[features])
  target_weighted = target_scaled * weights

STEP 3: Query KNN
  distances, indices = model.kneighbors(target_weighted, n_neighbors=30)

STEP 4: Filter results
  For each neighbor:
    SKIP if exact same player-season (the query itself)
    SKIP if exclude_self and same player (different season)
    SKIP if position not compatible

STEP 5: Calculate match score
  # Exponential decay from cosine distance
  match_score = 100 * exp(-distance * 5)
  match_score = clamp(match_score, 0, 100)

STEP 6: Return top 5 results

OUTPUT:
  List of dicts with:
  - Player, Season, PLAYER_ID
  - Stats (PTS, REB, AST, etc.)
  - MatchScore, Distance
  - Position
```

### Score Interpretation

| Cosine Distance | Match Score | Meaning |
|-----------------|-------------|---------|
| 0.02 | 90% | Almost identical statistically |
| 0.05 | 78% | Very similar |
| 0.10 | 61% | Similar |
| 0.15 | 47% | Somewhat similar |
| 0.20 | 37% | Different |
| 0.30 | 22% | Very different |

---

## Fuzzy Name Matching

### Purpose
Match player names between datasets that may have different formats.

### Location
`data_processing.py` -> `fuzzy_match_players()`

### Algorithm

```
INPUT:
  - name1: Name from dataset 1 (e.g., LEBRON data)
  - name2: Name from dataset 2 (e.g., contract data)

STEP 1: Normalize names
  - Convert to lowercase
  - Remove suffixes: Jr., Jr, Sr., III, II, IV
  - Handle prefixes: De', Van, La, Le (keep attached)
  - Remove punctuation except apostrophes

STEP 2: Split into first/last
  parts = name.split()
  first = parts[0]
  last = parts[-1] if len(parts) > 1 else ''

STEP 3: Calculate similarity
  Using rapidfuzz library:
  first_score = fuzz.ratio(first1, first2)
  last_score = fuzz.ratio(last1, last2)

STEP 4: Apply thresholds
  MATCH if:
    - first_score >= 95 AND last_score >= 85
    - OR overall_score >= 90

OUTPUT:
  - Boolean match result
  - Match score (0-100)
```

### Examples

| Name 1 | Name 2 | Match? | Score |
|--------|--------|--------|-------|
| "Shai Gilgeous-Alexander" | "Shai Gilgeous-Alexander" | Yes | 100 |
| "PJ Washington" | "P.J. Washington" | Yes | 98 |
| "Jaren Jackson Jr." | "Jaren Jackson" | Yes | 95 |
| "Nicolas Claxton" | "Nic Claxton" | Yes | 92 |
| "De'Aaron Fox" | "DeAaron Fox" | Yes | 96 |
| "LeBron James" | "Lebron James" | Yes | 100 |

