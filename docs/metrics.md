# Metrics Reference

Complete reference for all metrics used in Sieve.

---

## Primary Metrics

### LEBRON

**Full Name:** Luck-adjusted player Estimate using a Box prior On/off Rating Normalized

**Source:** BBall Index

**What it measures:** Overall player impact per 100 possessions

**Scale:** -5 to +10 (league average = 0)

**Formula:** Proprietary (BBall Index). Combines:
- On/off court net rating differential
- Box score regression (stabilizes small samples)
- Luck adjustment (removes 3PT variance, etc.)

**Interpretation:**

| Score | Tier | Description |
|-------|------|-------------|
| +6 to +10 | MVP | Top 5-10 players in league |
| +4 to +6 | All-NBA | Elite players |
| +2 to +4 | All-Star | Very good starters |
| +1 to +2 | Quality Starter | Above average |
| 0 to +1 | Average | League average impact |
| -1 to 0 | Below Average | Rotation player |
| -3 to -1 | Poor | Limited rotation |
| -5 to -3 | Negative | Hurts team when playing |

**Why we use it:** Single best publicly available all-in-one metric. Captures both box score production and on/off impact.

---

### O-LEBRON

**What it measures:** Offensive component of LEBRON

**Scale:** -5 to +8

**Interpretation:**
- +4 or higher: Elite offensive player
- +2 to +4: Good offensive player
- 0 to +2: Average offense
- Negative: Below average offense

---

### D-LEBRON

**What it measures:** Defensive component of LEBRON

**Scale:** -3 to +5

**Interpretation:**
- +3 or higher: Elite defender
- +1 to +3: Good defender
- 0 to +1: Average defense
- Negative: Defensive liability

**Note:** Defensive metrics are harder to measure than offensive. D-LEBRON has more noise.

---

### LEBRON WAR

**Full Name:** Wins Above Replacement

**What it measures:** Cumulative wins added compared to replacement-level player

**Scale:** -2 to +20 per season

**Formula:** LEBRON * minutes_factor

**Interpretation:**
- 15+: MVP candidate
- 10-15: All-NBA caliber
- 5-10: All-Star caliber
- 2-5: Quality starter
- 0-2: Rotation player
- Negative: Below replacement level

**Difference from LEBRON:** LEBRON is rate stat (per 100 possessions), WAR is cumulative (total value).

---

## Calculated Metrics

### value_gap

**What it measures:** How much value a player provides relative to their salary

**Calculated in:** `data_processing.calculate_player_value_metrics()`

**Formula:**
```python
# Step 1: Normalize salary to 0-100 percentile
salary_min = df['current_year_salary'].min()
salary_max = df['current_year_salary'].max()
salary_norm = 100 * (salary - salary_min) / (salary_max - salary_min)

# Step 2: Normalize LEBRON to 0-100 percentile
lebron_min = df['LEBRON'].min()
lebron_max = df['LEBRON'].max()
impact_norm = 100 * (LEBRON - lebron_min) / (lebron_max - lebron_min)

# Step 3: Calculate value gap
value_gap = impact_norm * 1.4 - salary_norm * 0.9 - 10
```

**Why these weights:**
- **1.4 on impact:** On-court contribution matters more than cost
- **0.9 on salary:** Slight discount because salary has larger variance
- **-10 offset:** Centers distribution so average player is near 0

**Scale:** Typically -50 to +80

**Interpretation:**

| value_gap | Meaning | Examples |
|-----------|---------|----------|
| +40 or higher | Extreme value | Rookie stars on rookie deals |
| +20 to +40 | Great value | Good players on cheap contracts |
| +5 to +20 | Slight value | Fairly paid or better |
| -5 to +5 | Fair | Paid appropriately |
| -20 to -5 | Slight overpay | Paid more than production |
| -40 to -20 | Overpaid | Significant overpay |
| Below -40 | Extreme overpay | Bad contracts |

**Example Calculations:**

| Player | LEBRON | Salary | salary_norm | impact_norm | value_gap |
|--------|--------|--------|-------------|-------------|-----------|
| SGA | 8.67 | $38M | 70 | 100 | 100*1.4 - 70*0.9 - 10 = **67** |
| Jokic | 8.06 | $55M | 100 | 93 | 93*1.4 - 100*0.9 - 10 = **30.2** |
| Harden | 2.15 | $39M | 72 | 52 | 52*1.4 - 72*0.9 - 10 = **-12** |
| Rookie Star | 5.0 | $5M | 5 | 70 | 70*1.4 - 5*0.9 - 10 = **83.5** |

---

### salary_norm

**What it measures:** Player's salary as a percentile (0-100)

**Formula:** `100 * (salary - min) / (max - min)`

**Note:** Calculated relative to the CURRENT FILTER. If you filter to only max players, a $40M salary might be 0th percentile.

---

### impact_norm

**What it measures:** Player's LEBRON as a percentile (0-100)

**Formula:** `100 * (LEBRON - min) / (max - min)`

**Note:** Same filter dependency as salary_norm.

---

## NBA API Advanced Stats

### USG_PCT (Usage Percentage)

**What it measures:** Percentage of team possessions "used" by player while on floor

**Formula:**
```
100 * ((FGA + 0.44 * FTA + TOV) * (Team_Minutes / 5)) / (Player_Minutes * (Team_FGA + 0.44 * Team_FTA + Team_TOV))
```

**Scale:** 10% to 40%

**Interpretation:**

| USG% | Role |
|------|------|
| 30%+ | Primary option (ball dominant) |
| 25-30% | Secondary option |
| 20-25% | Tertiary option |
| 15-20% | Role player |
| <15% | Low usage role |

**Why it matters for Diamond Finder:** USG% is the #1 indicator of a player's offensive role. Two Shot Creators with 30% USG play more similarly than one with 30% and one with 18%.

---

### AST_PCT (Assist Percentage)

**What it measures:** Percentage of teammate field goals assisted while player on floor

**Formula:**
```
100 * AST / (((Minutes / (Team_Minutes / 5)) * Team_FGM) - Player_FGM)
```

**Scale:** 5% to 50%

**Interpretation:**

| AST% | Role |
|------|------|
| 35%+ | Primary playmaker (elite) |
| 25-35% | Primary playmaker |
| 15-25% | Secondary playmaker |
| 10-15% | Limited playmaking |
| <10% | Non-playmaker |

---

### TS_PCT (True Shooting Percentage)

**What it measures:** Shooting efficiency accounting for 2P, 3P, and FT

**Formula:**
```
PTS / (2 * (FGA + 0.44 * FTA))
```

**Why 0.44?** Free throws don't always use a possession (and-ones, technicals).

**Scale:** 45% to 70%

**Interpretation:**

| TS% | Efficiency |
|-----|------------|
| 65%+ | Elite |
| 60-65% | Very good |
| 57-60% | Above average |
| 54-57% | Average |
| 50-54% | Below average |
| <50% | Poor |

**League average:** ~57%

---

### DEF_RATING (Defensive Rating)

**What it measures:** Points allowed per 100 possessions while player on floor

**Scale:** 100 to 125

**Interpretation:** LOWER IS BETTER

| DEF_RATING | Quality |
|------------|---------|
| <105 | Elite defense |
| 105-110 | Good defense |
| 110-115 | Average |
| 115-120 | Below average |
| >120 | Poor defense |

**Caveat:** Highly influenced by teammates. A good defender on a bad team will have poor DEF_RATING.

---

### REB_PCT (Rebound Percentage)

**What it measures:** Percentage of available rebounds grabbed while on floor

**Scale:** 5% to 25%

**Interpretation:**
- 20%+: Elite rebounder (usually center)
- 15-20%: Good rebounder
- 10-15%: Average
- <10%: Poor rebounder

---

### OREB_PCT (Offensive Rebound Percentage)

**What it measures:** Percentage of offensive rebounds grabbed

**Scale:** 1% to 15%

**Why it matters:** High OREB% indicates a "big man" playstyle (crashes boards).

---

## Diamond Finder Similarity Score

**What it measures:** How similarly two players play (0-100%)

**Components:**

1. **Base Score (0-60):** From KNN cosine distance
   ```python
   normalized_dist = (distance - min_dist) / (max_dist - min_dist)
   base_score = 60 * (1 - normalized_dist)
   ```

2. **Archetype Bonus (-10 to +20):**
   - Exact archetype match: +20
   - Same archetype family: +8
   - Different archetype: -10

3. **Advanced Stats Bonus (0-15):**
   - USG within 3%: +5
   - AST% within 5%: +5
   - TS% within 3%: +5

**Total:** base_score + archetype_bonus + stats_bonus (clamped 30-98)

**Interpretation:**

| Score | Meaning |
|-------|---------|
| 90%+ | Excellent match - very similar playstyle |
| 75-90% | Good match - similar role |
| 60-75% | Decent match - some similarities |
| 45-60% | Fair match - different styles |
| <45% | Poor match - no real comparison |

---

## Historical Similarity Score

**What it measures:** Statistical similarity to historical player-seasons

**Formula:**
```python
match_score = 100 * exp(-cosine_distance * 5)
```

**Example conversions:**

| Cosine Distance | Match Score |
|-----------------|-------------|
| 0.02 | 90% |
| 0.05 | 78% |
| 0.10 | 61% |
| 0.15 | 47% |
| 0.20 | 37% |
| 0.30 | 22% |

**Interpretation:** Same as Diamond Finder scores.

