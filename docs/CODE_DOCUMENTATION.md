# Sieve Codebase Documentation

This document provides a detailed technical breakdown of the Sieve application. It explains the purpose of each file, the logic behind key functions, and how the components interact.

---

## 1. Architecture Overview

The application follows a modular **Model-View-Controller (MVC)** pattern adapted for Dash:

*   **Controller (`dashboard.py`)**: Entry point. Handles app initialization, data loading, and user interaction (callbacks).
*   **View (`visualizations.py`)**: Generates all Plotly charts and Dash tables.
*   **Structure (`layout.py`)**: Defines the HTML/CSS layout of the application.
*   **Model (`data_processing.py`)**: Handles data ingestion, cleaning, merging, and metric calculations.
*   **Ingestion (`scraper.py`)**: Fetches raw data from external sources.

---

## 2. File Breakdown

### `dashboard.py` (The Controller)
**Purpose**: The main entry point for the web application. It ties together the data, layout, and visualizations.

**Key Responsibilities**:
1.  **Initialization**: Sets up the Dash app and Flask server.
2.  **Data Loading**: Calls `data_processing` functions to load `df` (players) and `df_teams`.
3.  **Layout Assembly**: Calls `layout.py` functions to build the UI.
4.  **Callbacks**: Defines the `update_dashboard` function which reacts to user input (sliders) and updates the charts.

**Main Functions**:
*   `enforce_salary_constraints(min_val, max_val)`: A callback that ensures the minimum salary slider never exceeds the maximum slider.
*   `update_dashboard(min_lebron, min_salary, max_salary)`: The core callback.
    *   **Input**: User filter values.
    *   **Logic**: Filters the global `df`, recalculates `Value Gap` for the filtered subset, and calls functions from `visualizations.py` to generate updated charts.
    *   **Output**: Returns 4 figures and 3 table contents to the frontend.

---

### `visualizations.py` (The View)
**Purpose**: Contains all logic for generating visual components. This keeps the main dashboard file clean.

**Key Responsibilities**:
*   Creating Plotly Graph Objects (`go.Figure`) for charts.
*   Creating Dash Bootstrap Tables (`dbc.Table`) for data display.

**Main Functions**:
*   `create_efficiency_quadrant(df_teams)`: Generates the scatter plot with the Z-score background gradient.
    *   **Logic**: Creates a meshgrid of Wins vs Payroll, calculates Z-scores for every point on the grid, and renders a contour plot (Green/Red) behind the team logos.
*   `create_team_grid(df_teams)`: Generates the 6x5 grid of team tiles.
    *   **Logic**: Sorts teams by `Efficiency_Index`, assigns them (x, y) coordinates, and plots them as equal-sized squares colored by efficiency.
*   `create_salary_impact_scatter(filtered_df)`: Plots Salary (Y) vs LEBRON Impact (X).
*   `create_age_impact_scatter(filtered_df)`: Plots LEBRON Impact (Y) vs Age (X), with a "Peak Years" reference band (26-30).
*   `create_underpaid_bar(filtered_df)` / `create_overpaid_bar(filtered_df)`: Generates horizontal bar charts for the top 10 best/worst value players.

---

### `layout.py` (The Structure)
**Purpose**: Defines the HTML structure and CSS styling of the application.

**Key Responsibilities**:
*   Defining the Tab structure (Player Analysis vs Team Analysis).
*   Organizing Cards, Rows, and Columns.
*   Setting up the Filter controls (Sliders).

**Main Functions**:
*   `create_player_tab(df)`: Returns the layout for the Player Analysis tab, including the filter card and the 4-chart grid.
*   `create_team_tab(df_teams, ...)`: Returns the layout for the Team Analysis tab.
*   `create_main_layout(...)`: Wraps everything in the main container and header.

---

### `data_processing.py` (The Model)
**Purpose**: The "Brain" of the application. Handles all data manipulation.

**Key Responsibilities**:
*   Loading CSVs (`LEBRON.csv`, `contracts.csv`).
*   Merging datasets.
*   Calculating advanced metrics.

**Main Functions**:
*   `load_and_merge_data()`:
    *   Loads raw CSVs.
    *   Cleans player names.
    *   Merges on `player_name`.
    *   **Critical Logic**: Fills missing `current_year_salary` with `year_4` (Guaranteed amount) if available, to ensure data completeness.
*   `calculate_player_value_metrics(df)`:
    *   **Logic**: Normalizes Salary and LEBRON Impact to a 0-100 scale.
    *   **Metric**: `Value Gap = Impact_Norm - Salary_Norm`.
*   `calculate_team_metrics(df)`:
    *   Aggregates player data by Team.
    *   Fetches Standings (Wins/Losses).
    *   **Metric**: `Efficiency Index = Z(Wins) - Z(Payroll)`.
*   `add_team_logos(df)`: Uses `nba_api` to fetch official CDN URLs for team logos.

---

### `scraper.py` (Data Ingestion)
**Purpose**: A standalone script to fetch contract data from Basketball Reference.

**Key Responsibilities**:
*   Web scraping using **Selenium** (headless Chrome).
*   Parsing HTML tables with **BeautifulSoup**.

**Main Functions**:
*   `scrape_bball_ref()`:
    *   Navigates to `basketball-reference.com/contracts/players.html`.
    *   Parses the table headers dynamically to handle phantom columns.
    *   Extracts contract values for the next 4 years.
    *   **Logic**: Deduplicates players (keeping the first entry) and calculates `Total Contract Value` and `Average Annual Value`.

---

### `app.py` (Legacy/Standalone)
**Purpose**: A command-line version of the analysis tool. Useful for running quick reports without the web interface.

**Key Responsibilities**:
*   Runs the same data processing pipeline.
*   Prints text-based reports to the console (Top Earners, Efficiency Rankings).
*   Exports data to CSVs (`sieve_player_analysis.csv`, `sieve_team_efficiency.csv`).

---

## 3. Key Metric Definitions

### Value Gap (Player)
Quantifies how much a player outperforms their contract.
1.  **Normalize Salary**: $S_{norm} = \frac{Salary - Min}{Max - Min} \times 100$
2.  **Normalize Impact**: $I_{norm} = \frac{LEBRON - Min}{Max - Min} \times 100$
3.  **Calculate Gap**: $Gap = I_{norm} - S_{norm}$

*   **Example**: A player with 90th percentile impact but 50th percentile salary has a Value Gap of +40 (Highly Underpaid).

### Efficiency Index (Team)
Quantifies how efficiently a team buys wins.
1.  **Z-Score Wins**: $Z_W = \frac{Wins - \mu_W}{\sigma_W}$
2.  **Z-Score Payroll**: $Z_P = \frac{Payroll - \mu_P}{\sigma_P}$
3.  **Calculate Index**: $Index = (2.0 \times Z_W) - Z_P$

*   **Interpretation**:
    *   **High Positive**: Winning significantly more than average (Wins are weighted 2x).
    *   **High Negative**: Losing more than average or spending inefficiently.
    *   **Note**: We prioritize on-court success (Wins) over pure financial savings.
