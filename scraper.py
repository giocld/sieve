"""
Data Scraper for Sieve NBA Analytics.

This script fetches player contract data from Basketball Reference.
It uses Selenium to handle dynamic content and BeautifulSoup for parsing the HTML structure.
The extracted data is cleaned, deduplicated, and saved to a CSV file.
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

def scrape_bball_ref():
    """
    Main scraping function.
    
    Workflow:
    1. Configures a headless Chrome browser.
    2. Navigates to the Basketball Reference contracts page.
    3. Waits for the data table to load.
    4. Parses the HTML table to extract player names and contract values.
    5. Cleans and structures the data into a DataFrame.
    6. Saves the result to 'basketball_reference_contracts.csv'.
    """
    print("=" * 70)
    print("Basketball Reference Contracts Scraper")
    print("Target: https://www.basketball-reference.com/contracts/players.html")
    print("=" * 70)

    # 1. Setup Headless Chrome
    # We use headless mode to run the scraper without opening a visible browser window.
    # This is more efficient for automated tasks.
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

        # Get Headers
        # The table structure is complex with grouped headers.
        # We look for the last row in the 'thead' section to get the actual column names.
        tbody = table.find('tbody')
        
        headers_tr = table.find('thead').find_all('tr')[-1]
        actual_headers = [th.get_text(strip=True) for th in headers_tr.find_all('th')]  
        
        # Filter out empty headers
        headers_clean = [h for h in actual_headers if h]
        
        # Map columns by name to their indices
        # This makes the code robust to column reordering in the source table.
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
            # Skip header rows that repeat in the middle of the table
            if 'class' in row.attrs and 'thead' in row.attrs['class']:
                continue
            
            # Extract all cells (both 'th' and 'td' tags)
            cells = row.find_all(['td', 'th'])
            
            # Ensure we have enough cells to match our headers
            if len(cells) < len(headers_clean):
                continue
                
            row_text = [c.get_text(strip=True) for c in cells]
            
            # Skip rows that are just repeating headers
            if row_text[idx_player] == "Player":
                continue
            
            p_name = row_text[idx_player]
            
            # Clean currency strings to floats
            val_year_0 = clean_money(row_text[idx_25_26])
            val_year_1 = clean_money(row_text[idx_26_27])
            val_year_2 = clean_money(row_text[idx_27_28])
            val_guaranteed = clean_money(row_text[idx_guaranteed])
            
            entry = {
                'player_name': p_name,
                'year_4': val_guaranteed, # Using 'year_4' as a key for Guaranteed amount
                'year_0': val_year_0,
                'year_1': val_year_1,
                'year_2': val_year_2
            }
            all_data.append(entry)
            
    except Exception as e:
        print(f"Fatal Error: {e}")
    finally:
        driver.quit()

    # Process Data
    if not all_data:
        print("No data collected")
        return

    df = pd.DataFrame(all_data)
    
    # Remove duplicates (Basketball Reference shows some players multiple times)
    # We keep the first occurrence which is usually the most up-to-date or primary entry.
    print(f"Extracted {len(df)} total rows")
    df = df.drop_duplicates(subset=['player_name'], keep='first')
    print(f"After deduplication: {len(df)} unique players")
    
    # 1. Calculate Contract Length
    # Logic: Base 1 if Guaranteed > 0, plus 1 for each specific year > 0
    def calc_length(row):
        count = 0
        if row['year_4'] > 0: count = 1
        if row['year_0'] > 0: count += 1
        if row['year_1'] > 0: count += 1
        if row['year_2'] > 0: count += 1
        return count

    df['contract_length'] = df.apply(calc_length, axis=1)

    # 2. Calculate Total Contract Value
    df['total_contract_value'] = df['year_4'] + df['year_0'] + df['year_1'] + df['year_2']
    
    # 3. Calculate Average Annual Value (AAV)
    df['average_annual_value'] = df.apply(
        lambda x: x['total_contract_value'] / x['contract_length'] if x['contract_length'] > 0 else 0, 
        axis=1
    )

    # 4. Set Current Year Salary
    df['current_year_salary'] = df['year_0']

    # 5. Calculate Years Remaining
    df['years_remaining'] = df['contract_length'].apply(lambda x: max(0, x - 1))
    
    # Sort by total value descending
    df = df.sort_values('total_contract_value', ascending=False)

    # Clean for Output (Float -> String for readability in CSV, 0 -> Empty)
    output_cols = [
        'player_name', 'year_4', 'year_0', 'year_1', 'year_2', 
        'contract_length', 'total_contract_value', 'average_annual_value', 
        'current_year_salary', 'years_remaining'
    ]
    
    output_df = df[output_cols].copy()
    
    cols_to_blank = ['year_4', 'year_0', 'year_1', 'year_2', 'current_year_salary']
    for col in cols_to_blank:
        output_df[col] = output_df[col].apply(lambda x: str(x) if x > 0 else "")

    print(f"\n✓ Extracted {len(output_df)} players")
    print(output_df.head(10).to_string(index=False))
    
    output_df.to_csv('basketball_reference_contracts.csv', index=False)

if __name__ == "__main__":
    scrape_bball_ref()