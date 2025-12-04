"""
Data Scraper for Sieve NBA Analytics.

This module provides two methods to get player contract data from Basketball Reference:
1. parse_bbref_csv() - Parse a manually downloaded CSV (recommended, includes player IDs)
2. scrape_bball_ref() - Web scrape using Selenium (fallback)

The CSV method is preferred because it includes BBRef player IDs for accurate matching.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os


def clean_money(money_str):
    """
    Converts a currency string (e.g., '$50,000') into a float (50000.0).
    
    Args:
        money_str (str): The string containing the currency value.
        
    Returns:
        float: The numeric value of the currency, or 0.0 if invalid.
    """
    if not money_str or str(money_str).strip() in ['-', '', '—', 'nan', 'None']:
        return 0.0
    clean = re.sub(r'[$,]', '', str(money_str)).strip()
    try:
        return float(clean)
    except ValueError:
        return 0.0


def parse_bbref_csv(csv_path='data/bbref_contracts_raw.csv', output_path='data/basketball_reference_contracts.csv'):
    """
    Parse a CSV exported from Basketball Reference contracts page.
    
    This is the PREFERRED method because it includes BBRef player IDs
    which enable accurate player matching across data sources.
    
    How to get the CSV:
    1. Go to https://www.basketball-reference.com/contracts/players.html
    2. Click "Share & Export" -> "Get table as CSV"
    3. Save as data/bbref_contracts_raw.csv
    4. Run: python -m src.scraper --csv
    
    Expected CSV format (first 2 rows are headers):
    ,,,Salary,Salary,Salary,Salary,Salary,Salary,,-additional
    Rk,Player,Tm,2025-26,2026-27,2027-28,2028-29,2029-30,2030-31,Guaranteed,bbref_id
    1,Stephen Curry,GSW,$59606817,$62587158,,,,,$122193975,curryst01
    
    Args:
        csv_path: Path to the raw BBRef CSV
        output_path: Where to save the processed CSV
        
    Returns:
        pd.DataFrame: Processed contract data
    """
    print("=" * 70)
    print("Basketball Reference CSV Parser")
    print(f"Input: {csv_path}")
    print("=" * 70)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        print("\nTo get this file:")
        print("1. Go to https://www.basketball-reference.com/contracts/players.html")
        print("2. Click 'Share & Export' -> 'Get table as CSV'")
        print(f"3. Save as {csv_path}")
        return None
    
    # Read CSV, skipping the first header row (the grouped headers)
    # The actual column names are on row 2
    df = pd.read_csv(csv_path, skiprows=1)
    
    print(f"Raw columns: {df.columns.tolist()}")
    print(f"Raw rows: {len(df)}")
    
    # Identify columns - BBRef uses varying column names
    # Find salary columns (they contain year patterns like 2025-26)
    salary_cols = [c for c in df.columns if re.match(r'\d{4}-\d{2}', str(c))]
    
    # Map columns
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == 'player':
            col_map['player_name'] = col
        elif col_lower == 'tm':
            col_map['team'] = col
        elif col_lower == 'guaranteed':
            col_map['guaranteed'] = col
        elif 'additional' in col_lower or col_lower == 'rk':
            # This is likely the BBRef ID column (last column) or rank
            if 'additional' in col_lower:
                col_map['bbref_id'] = col
    
    # The BBRef ID is often in a column named with '-additional' or similar
    # Let's also check the last column
    last_col = df.columns[-1]
    if 'bbref_id' not in col_map and last_col not in salary_cols and last_col.lower() != 'guaranteed':
        col_map['bbref_id'] = last_col
    
    print(f"Column mapping: {col_map}")
    print(f"Salary columns: {salary_cols}")
    
    # Build output DataFrame
    all_data = []
    
    for idx, row in df.iterrows():
        # Skip header rows that might be repeated
        player_name = str(row.get(col_map.get('player_name', 'Player'), ''))
        if not player_name or player_name.lower() == 'player' or player_name == 'nan':
            continue
        
        # Get BBRef ID
        bbref_id = ''
        if 'bbref_id' in col_map:
            bbref_id = str(row.get(col_map['bbref_id'], '')).strip()
            if bbref_id == 'nan':
                bbref_id = ''
        
        # Get team
        team = str(row.get(col_map.get('team', 'Tm'), '')).strip()
        if team == 'nan':
            team = ''
        
        # Get salary values
        salaries = {}
        for sal_col in salary_cols:
            salaries[sal_col] = clean_money(row.get(sal_col, 0))
        
        # Get guaranteed
        guaranteed = clean_money(row.get(col_map.get('guaranteed', 'Guaranteed'), 0))
        
        # Determine current year salary (first salary column with value)
        current_salary = 0
        year_0 = year_1 = year_2 = 0
        
        if salary_cols:
            if len(salary_cols) > 0:
                year_0 = salaries.get(salary_cols[0], 0)
                current_salary = year_0
            if len(salary_cols) > 1:
                year_1 = salaries.get(salary_cols[1], 0)
            if len(salary_cols) > 2:
                year_2 = salaries.get(salary_cols[2], 0)
        
        entry = {
            'player_name': player_name,
            'team': team,
            'bbref_id': bbref_id,
            'year_0': year_0,
            'year_1': year_1,
            'year_2': year_2,
            'year_4': guaranteed,  # Keep year_4 for backward compatibility
        }
        all_data.append(entry)
    
    if not all_data:
        print("No data extracted")
        return None
    
    df_out = pd.DataFrame(all_data)
    
    # Remove duplicates, keeping first (highest salary usually)
    print(f"Extracted {len(df_out)} total rows")
    df_out = df_out.drop_duplicates(subset=['player_name'], keep='first')
    print(f"After deduplication: {len(df_out)} unique players")
    
    # Calculate derived fields
    df_out['contract_length'] = df_out.apply(
        lambda r: sum([1 for v in [r['year_0'], r['year_1'], r['year_2']] if v > 0]),
        axis=1
    )
    
    df_out['total_contract_value'] = df_out['year_0'] + df_out['year_1'] + df_out['year_2']
    
    df_out['average_annual_value'] = df_out.apply(
        lambda x: x['total_contract_value'] / x['contract_length'] if x['contract_length'] > 0 else 0,
        axis=1
    )
    
    df_out['current_year_salary'] = df_out['year_0']
    df_out['years_remaining'] = df_out['contract_length'].apply(lambda x: max(0, x - 1))
    
    # Sort by salary descending
    df_out = df_out.sort_values('current_year_salary', ascending=False)
    
    # Output columns
    output_cols = [
        'player_name', 'team', 'bbref_id',
        'year_0', 'year_1', 'year_2', 'year_4',
        'contract_length', 'total_contract_value', 'average_annual_value',
        'current_year_salary', 'years_remaining'
    ]
    
    df_final = df_out[output_cols].copy()
    
    # Save
    df_final.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(df_final)} players to {output_path}")
    print(f"Players with BBRef IDs: {(df_final['bbref_id'] != '').sum()}")
    print("\nTop 10 by salary:")
    print(df_final[['player_name', 'team', 'bbref_id', 'current_year_salary']].head(10).to_string(index=False))
    
    return df_final


def scrape_bball_ref():
    """
    Web scrape contracts from Basketball Reference using Selenium.
    
    This is the FALLBACK method. Prefer parse_bbref_csv() when possible
    because it includes BBRef player IDs.
    
    Workflow:
    1. Configures a headless Chrome browser.
    2. Navigates to the Basketball Reference contracts page.
    3. Waits for the data table to load.
    4. Parses the HTML table to extract player names and contract values.
    5. Cleans and structures the data into a DataFrame.
    6. Saves the result to 'basketball_reference_contracts.csv'.
    """
    print("=" * 70)
    print("Basketball Reference Contracts Scraper (Selenium)")
    print("Target: https://www.basketball-reference.com/contracts/players.html")
    print("=" * 70)
    print("\nNote: For better data (with player IDs), use parse_bbref_csv() instead.")

    # 1. Setup Headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    url = "https://www.basketball-reference.com/contracts/players.html"
    
    all_data = []
    
    try:
        print(f"Loading: {url}")
        driver.get(url)
        
        # Wait for the specific table id 'player-contracts' to ensure data is loaded
        wait = WebDriverWait(driver, 15)
        try:
            wait.until(EC.presence_of_element_located((By.ID, "player-contracts")))
            print("Table loaded")
        except:
            print("Timeout waiting for table #player-contracts")
            return

        # Get full page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', id='player-contracts')
        
        if not table:
            print("Table not found in parsed HTML")
            return

        tbody = table.find('tbody')
        
        headers_tr = table.find('thead').find_all('tr')[-1]
        actual_headers = [th.get_text(strip=True) for th in headers_tr.find_all('th')]  
        headers_clean = [h for h in actual_headers if h]
        
        # Map columns by name
        try:
            idx_player = headers_clean.index("Player")
            idx_25_26 = headers_clean.index("2025-26")
            idx_26_27 = headers_clean.index("2026-27")
            idx_27_28 = headers_clean.index("2027-28")
            idx_guaranteed = headers_clean.index("Guaranteed")
            
        except ValueError as e:
            print(f"Column mapping failed. Headers found: {headers_clean}")
            print(f"Error: {e}")
            return

        # Extract Rows
        rows = tbody.find_all('tr')
        print(f"Found {len(rows)} rows")
        
        for row in rows:
            if 'class' in row.attrs and 'thead' in row.attrs['class']:
                continue
            
            cells = row.find_all(['td', 'th'])
            
            if len(cells) < len(headers_clean):
                continue
                
            row_text = [c.get_text(strip=True) for c in cells]
            
            if row_text[idx_player] == "Player":
                continue
            
            p_name = row_text[idx_player]
            
            # Try to get BBRef ID from the player link
            bbref_id = ''
            player_cell = cells[idx_player]
            player_link = player_cell.find('a')
            if player_link and 'href' in player_link.attrs:
                # Extract ID from URL like /players/c/curryst01.html
                href = player_link['href']
                match = re.search(r'/players/\w/(\w+)\.html', href)
                if match:
                    bbref_id = match.group(1)
            
            val_year_0 = clean_money(row_text[idx_25_26])
            val_year_1 = clean_money(row_text[idx_26_27])
            val_year_2 = clean_money(row_text[idx_27_28])
            val_guaranteed = clean_money(row_text[idx_guaranteed])
            
            entry = {
                'player_name': p_name,
                'bbref_id': bbref_id,
                'year_4': val_guaranteed,
                'year_0': val_year_0,
                'year_1': val_year_1,
                'year_2': val_year_2
            }
            all_data.append(entry)
            
    except Exception as e:
        print(f"Fatal Error: {e}")
    finally:
        driver.quit()

    if not all_data:
        print("No data collected")
        return

    df = pd.DataFrame(all_data)
    
    print(f"Extracted {len(df)} total rows")
    df = df.drop_duplicates(subset=['player_name'], keep='first')
    print(f"After deduplication: {len(df)} unique players")
    print(f"Players with BBRef IDs: {(df['bbref_id'] != '').sum()}")
    
    # Calculate derived fields
    def calc_length(row):
        count = 0
        if row['year_4'] > 0: count = 1
        if row['year_0'] > 0: count += 1
        if row['year_1'] > 0: count += 1
        if row['year_2'] > 0: count += 1
        return count

    df['contract_length'] = df.apply(calc_length, axis=1)
    df['total_contract_value'] = df['year_4'] + df['year_0'] + df['year_1'] + df['year_2']
    
    df['average_annual_value'] = df.apply(
        lambda x: x['total_contract_value'] / x['contract_length'] if x['contract_length'] > 0 else 0, 
        axis=1
    )

    df['current_year_salary'] = df['year_0']
    df['years_remaining'] = df['contract_length'].apply(lambda x: max(0, x - 1))
    
    df = df.sort_values('total_contract_value', ascending=False)

    output_cols = [
        'player_name', 'bbref_id',
        'year_4', 'year_0', 'year_1', 'year_2', 
        'contract_length', 'total_contract_value', 'average_annual_value', 
        'current_year_salary', 'years_remaining'
    ]
    
    output_df = df[output_cols].copy()

    print(f"\nExtracted {len(output_df)} players")
    print(output_df.head(10).to_string(index=False))
    
    output_df.to_csv('data/basketball_reference_contracts.csv', index=False)


if __name__ == "__main__":
    import sys
    
    if '--csv' in sys.argv:
        # Use CSV parser (preferred)
        csv_path = 'data/bbref_contracts_raw.csv'
        if len(sys.argv) > 2 and not sys.argv[-1].startswith('-'):
            csv_path = sys.argv[-1]
        parse_bbref_csv(csv_path)
    else:
        # Use Selenium scraper (fallback)
        scrape_bball_ref()
