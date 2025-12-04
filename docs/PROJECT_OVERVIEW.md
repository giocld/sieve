# Sieve NBA Analytics - Project Overview

## Introduction
Sieve is an advanced NBA analytics dashboard designed to identify market inefficiencies in player contracts and team spending. By synthesizing performance metrics with financial data, the platform provides unique insights into player value, team efficiency, and historical player comparisons.

## Technology Stack
The application is built using a modern Python data science stack:

*   **Dash (by Plotly)**: The core web framework used to build the interactive dashboard. It allows us to create complex, reactive web applications using only Python.
*   **Plotly Graph Objects**: Used for creating high-fidelity, interactive visualizations (scatter plots, radar charts, heatmaps).
*   **Pandas**: The backbone of our data manipulation. We use it for cleaning, merging, and aggregating large datasets of player stats and contracts.
*   **Scikit-Learn**: Powers our "Similarity Engine". We use the `NearestNeighbors` algorithm and `StandardScaler` to normalize data and find statistical doppelgängers.
*   **NBA API**: A Python client for the official NBA Stats API, used to fetch real-time and historical data.
*   **Selenium & BeautifulSoup**: Used in our custom scraper to harvest contract data from third-party websites that don't offer public APIs.

## Codebase Structure
The project is organized into modular components to separate concerns:

### 1. Core Application (`src/dashboard.py`)
This is the entry point of the application. It initializes the Dash app, loads the data, and defines the **Callbacks**—the logic that updates the charts when a user interacts with the UI (e.g., moving a slider).

### 2. Data Logic (`src/data_processing.py`)
This module handles all the "heavy lifting" for data:
*   **Loading & Merging**: Joins the LEBRON impact data with the Contract data.
*   **Calculations**: Contains the formulas for "Value Gap" and "Efficiency Index".
*   **Machine Learning**: Builds and trains the K-Nearest Neighbors model for player comparisons.
*   **API Fetching**: Handles requests to the NBA API for team stats and logos.

### 3. Visualization Engine (`src/visualizations.py`)
This file contains the code to generate every chart in the app. It takes raw DataFrames as input and returns Plotly `Figure` objects. This separation allows us to test charts independently of the web app.

### 4. Layout & UI (`src/layout.py`)
Defines the HTML structure of the dashboard using Dash Bootstrap Components. It organizes the app into tabs (Player, Team, Similarity) and handles the responsive grid system for mobile compatibility.

### 5. Scraper (`src/scraper.py`)
A standalone script that runs independently to update our contract database. It navigates to Basketball Reference, parses the contract tables, and saves the clean data to a CSV.

## Data Flow Architecture
1.  **Ingestion**:
    *   `scraper.py` runs to fetch the latest contract data.
    *   `dashboard.py` calls the NBA API to get fresh team standings and stats.
2.  **Processing**:
    *   `data_processing.py` merges these disparate sources into a single "Master DataFrame".
    *   It calculates derived metrics (e.g., `Value Gap = Impact - Salary`).
3.  **Modeling**:
    *   The Similarity Engine normalizes historical player stats and fits a KNN model to the data.
4.  **Visualization**:
    *   The processed data is passed to `visualizations.py` to create charts.
    *   `layout.py` arranges these charts into a user-friendly grid.

## Methodology & Calculations

### 1. The "Value Gap" (Player Valuation)
The core of our player analysis is the **Value Gap**, a proprietary metric that quantifies the difference between a player's on-court production and their financial cost.

*   **How it works**: We normalize both the player's Impact Score (LEBRON) and their Salary onto a comparable 0-100 scale.
*   **The Calculation**: We subtract the Normalized Salary from the Normalized Impact (with specific weightings to prioritize on-court performance).
*   **Interpretation**:
    *   **Positive Gap**: The player is **Underpaid** (delivering more wins than their salary suggests).
    *   **Negative Gap**: The player is **Overpaid** (costing more than their production warrants).

### 2. Team Efficiency Index
We evaluate front-office performance using the **Efficiency Index**, which measures how effectively a team translates payroll dollars into wins.

*   **How it works**: We compare every team to the league average using statistical Z-scores.
*   **The Calculation**: We weigh "Wins Above Average" twice as heavily as "Payroll Below Average."
*   **Result**: This highlights teams that win significantly while spending less (e.g., young, rebuilding contenders) versus teams that spend heavily but underperform.

### 3. Similarity Engine (Player Comparison)
To compare players across eras, we built a **Similarity Engine** using Machine Learning (K-Nearest Neighbors).

*   **Features**: It analyzes over 20 distinct attributes, including shooting efficiency, defensive versatility, playmaking style, and ball-dominance (tracking data).
*   **Logic**: It uses "Cosine Similarity" to find players with matching *styles* of play, rather than just matching raw stat totals. This allows us to find the "modern-day equivalent" of historical players.
