# Sieve - NBA Archetype Salary Analysis

## Overview
Sieve is a Python project that analyzes NBA player salaries relative to their on-court impact based on archetype classifications. It combines statistical performance metrics (LEBRON and WAR metrics) with contract data scraped from Basketball Reference to identify value gaps, players who may be overpaid or underpaid compared to their impact.

## Features
- Scrapes NBA player contract data from Basketball Reference.
- Integrates with existing player impact data (LEBRON metrics).
- Merges and processes player salary and impact data.
- Calculates normalized salary and impact scores.
- Highlights top underpaid and overpaid players by archetype.
- Interactive dashboard to explore salary vs impact with visualizations.
- Exports detailed player salary and impact analysis.

## Project Structure
- `scraper.py`: Script to scrape player contract data from Basketball Reference using Selenium and BeautifulSoup.
- `app.py`: Main analysis pipeline that loads data, merges statistics and contracts, runs the salary impact analysis, and exports results.
- `dashboard.py`: Dash-based interactive web dashboard for visualizing the analysis with sliders, charts, and tables.
- `LEBRON.csv`: Player impact data with LEBRON/WAR metrics and archetype classifications.
- `basketball_reference_contracts.csv`: Scraped NBA contract data including salary and contract length.
- `sieve_analysis.csv`: Output CSV with merged and analyzed salary impact data.
- `venv/`: Python virtual environment with required dependencies (not tracked in Git).

## Installation
1. Clone the repository.

2. Create a Python virtual environment with these dependencies:
     pip install pandas numpy selenium beautifulsoup4 dash plotly dash-bootstrap-components
3. Run the `dashboard.py` file

![Dashboard screenshot](/screenshots/screenshot1.jpg)


![Dashboard screenshot](/screenshots/screenshot2.jpg)
