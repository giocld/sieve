# Data & Metrics Reference

## 1. Codebase Guide

### File Responsibilities
| File Path | Primary Responsibility | Key Libraries Used |
| :--- | :--- | :--- |
| `src/dashboard.py` | **App Controller**: Initializes the app, handles user interactions (callbacks), and coordinates data flow. | `dash`, `pandas` |
| `src/data_processing.py` | **Data Logic**: Loads CSVs, merges datasets, calculates metrics, and trains the ML model. | `pandas`, `sklearn`, `nba_api` |
| `src/visualizations.py` | **Chart Factory**: Contains functions to generate every Plotly chart and Dash table. | `plotly.graph_objects`, `dash_table` |
| `src/layout.py` | **UI Structure**: Defines the HTML/CSS layout, tabs, and component arrangement. | `dash_bootstrap_components` |
| `src/scraper.py` | **Data Scraper**: Fetches contract data from the web using headless Chrome. | `selenium`, `beautifulsoup4` |

### Key Functions
| Function Name | Defined In | Description |
| :--- | :--- | :--- |
| `calculate_player_value_metrics(df)` | `data_processing.py` | Adds `value_gap`, `salary_norm`, and `impact_norm` columns to the player DataFrame. |
| `calculate_team_metrics(df)` | `data_processing.py` | Aggregates player data to calculate Team Payroll, Total WAR, and the `Efficiency_Index`. |
| `build_similarity_model(df)` | `data_processing.py` | Prepares features, scales data, and trains the `NearestNeighbors` model for player comparisons. |
| `create_efficiency_quadrant(df)` | `visualizations.py` | Generates the scatter plot comparing Team Wins vs. Payroll with the background efficiency gradient. |
| `update_dashboard(...)` | `dashboard.py` | The main callback function that reacts to slider changes and updates all player charts. |

---

## 2. Data Dictionary

### Player Data (Merged)
| Column Name | Source | Description |
| :--- | :--- | :--- |
| `player_name` | Shared | Unique identifier for players. |
| `LEBRON` | LEBRON Data | **Impact Score**. Estimate of points added per 100 possessions. |
| `LEBRON WAR` | LEBRON Data | **Wins Above Replacement**. Cumulative impact on winning. |
| `current_year_salary` | Contract Data | The guaranteed salary for the current season (e.g., 2024-25). |
| `value_gap` | **Calculated** | The difference between normalized impact and normalized salary. |
| `archetype` | LEBRON Data | Combined role description (e.g., "Shot Creator / Wing Defender"). |

### Team Data (Aggregated)
| Column Name | Source | Description |
| :--- | :--- | :--- |
| `Abbrev` | NBA API | 3-letter team code (e.g., BOS, LAL). |
| `WINS` / `LOSSES` | NBA API | Current season record. |
| `Total_Payroll` | **Calculated** | Sum of `current_year_salary` for all players on the roster. |
| `Efficiency_Index` | **Calculated** | Z-Score metric: `(2.0 * Wins_Z) - Payroll_Z`. |
| `Cost_Per_Win` | **Calculated** | `Total_Payroll / WINS`. |

---

## 3. Key Metrics & Calculations

| Metric Name | Formula / Logic | Purpose |
| :--- | :--- | :--- |
| **Value Gap** | `(Impact Score * 1.4) - (Salary Score * 0.9) - 10` | Identifies market inefficiencies. <br> **Positive** = Underpaid (Good Value) <br> **Negative** = Overpaid (Bad Value) |
| **Efficiency Index** | `(2.0 * Wins_ZScore) - Payroll_ZScore` | Ranks teams by how efficiently they spend money. Rewards winning cheaply and penalizes losing expensively. |
| **Cost Per Win** | `Total Payroll / Total Wins` | Simple financial metric showing the dollar cost of each victory. |
| **Similarity Score** | Cosine Similarity (0-100%) | Measures how closely two players' statistical profiles match, regardless of era. |

---

## 4. Visualization Reference

| Chart Name | Type | Data Used | What It Shows |
| :--- | :--- | :--- | :--- |
| **Efficiency Quadrant** | Scatter Plot | X: Payroll <br> Y: Wins | Segments teams into 4 categories: **Elite** (High Wins/Low Pay), **Contenders**, **Rebuilding**, and **Disaster** (Low Wins/High Pay). |
| **Team Grid** | Heatmap Grid | Efficiency Index | A quick-glance leaderboard of the smartest front offices in the league. |
| **Salary vs. Impact** | Scatter Plot | X: LEBRON <br> Y: Salary | The "Market Map" of the NBA. Players above the trendline are overpaid; players below are bargains. |
| **Value Gap Bars** | Bar Chart | Value Gap | Top 20 lists for the best contracts (Underpaid) and worst contracts (Overpaid). |
| **Team Radar** | Radar Chart | Off/Def Ratings, Reb/Ast % | Compares two teams' strengths and weaknesses shape. |
