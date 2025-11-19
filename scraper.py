from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def clean_money(money_str): #cleans the string from $x to just x which is a float
    """Cleans string '$50,000' -> 50000.0"""
    if not money_str or str(money_str).strip() in ['-', '', '—', 'nan', 'None']:
        return 0.0
    clean = re.sub(r'[$,]', '', str(money_str)).strip() #here it gets stripped
    try:
        return float(clean)
    except ValueError:
        return 0.0

def scrape_bball_ref():#scrapes bball ref for contract values
    print("=" * 70)
    print("Basketball Reference Contracts Scraper")
    print("Target: https://www.basketball-reference.com/contracts/players.html")
    print("=" * 70)

    # 1. Setup Headless Chrome, not using a GUI
    chrome_options = Options()
    chrome_options.add_argument("--headless")#no browser
    chrome_options.add_argument("--no-sandbox")#no os security model
    chrome_options.add_argument("--disable-dev-shm-usage") #more resources
    chrome_options.add_argument("--window-size=1920,1080")#sets window
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    #makes the driver with these confs
    driver = webdriver.Chrome(options=chrome_options)
    url = "https://www.basketball-reference.com/contracts/players.html"
    
    all_data = []#list to store
    
    try:
        print(f"Loading: {url}")
        driver.get(url)
        
        # Wait for the specific table id 'player-contracts'
        wait = WebDriverWait(driver, 15)
        try:
            wait.until(EC.presence_of_element_located((By.ID, "player-contracts")))
            print("Table loaded")
        except:
            print("Timeout waiting for table #player-contracts")
            return

        # Get full page source
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', id='player-contracts')
        
        if not table:
            print("Table not found in parsed HTML")
            return

        # Get Headers to map dynamically
        # The header row usually contains: Player, Team, 2025-26, 2026-27, 2027-28, 2028-29, 2029-30, 2030-31, Guaranteed
        thead = table.find('thead')
        headers = [th.get_text(strip=True) for th in thead.find_all('th')]
        
        # We need specific indices based on column names
        try:
            idx_player = headers.index("Player")
            # Adjust season columns based on what is actually visible on the site
            # Usually: [Rank, Player, Team, 2025-26, 2026-27, 2027-28, 2028-29, 2029-30, 2030-31, Guaranteed]
            # Target mapping based on your request:
            # year_0 = 2027-28
            # year_1 = 2028-29
            # year_2 = 2029-30
            # year_4 = Guaranteed
            
            idx_27_28 = headers.index("2027-28")
            idx_28_29 = headers.index("2028-29")
            idx_29_30 = headers.index("2029-30")
            idx_guaranteed = headers.index("Guaranteed")
            
        except ValueError as e:
            print(f"Column mapping failed. Headers found: {headers}")
            print(f"Error: {e}")
            return

        # Extract Rows
        tbody = table.find('tbody')
        rows = tbody.find_all('tr')
        print(f"Found {len(rows)} rows")
        
        for row in rows:
            # Skip header rows that repeat in the table
            if 'class' in row.attrs and 'thead' in row.attrs['class']:
                continue
            
            # bball-ref uses 'th' for the player name rank sometimes, but 'td' for data
            cells = row.find_all(['td', 'th'])
            
            # Ensure we have enough cells
            if len(cells) < len(headers):
                continue
                
            # Extract using indices
            # We usually convert all cells to text list first to match `headers` length
            row_text = [c.get_text(strip=True) for c in cells]
            
            # Check alignment
            if row_text[idx_player] == "Player": continue
            
            p_name = row_text[idx_player]
            # player name has tags strip noise
            
            val_year_0 = clean_money(row_text[idx_27_28])
            val_year_1 = clean_money(row_text[idx_28_29])
            val_year_2 = clean_money(row_text[idx_29_30])
            val_guaranteed = clean_money(row_text[idx_guaranteed])
            
            entry = {
                'player_name': p_name,
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

    # Process Data
    if not all_data:
        print("No data collected")
        return

    df = pd.DataFrame(all_data)
    
    
    
    # 1. Contract Length logic
    # Logic: Base 1 if Guaranteed > 0, plus 1 for each specific year > 0
    def calc_length(row):
        count = 0
        if row['year_4'] > 0: count = 1
        if row['year_0'] > 0: count += 1
        if row['year_1'] > 0: count += 1
        if row['year_2'] > 0: count += 1
        return count

    df['contract_length'] = df.apply(calc_length, axis=1)

    # 2. Total Contract Value (Sum of these specific columns)
    df['total_contract_value'] = df['year_4'] + df['year_0'] + df['year_1'] + df['year_2']
    
    # 3. Average Annual Value
    df['average_annual_value'] = df.apply(
        lambda x: x['total_contract_value'] / x['contract_length'] if x['contract_length'] > 0 else 0, 
        axis=1
    )

    # 4. Current Year Salary (Mapped to year_0 / 2027-28)
    df['current_year_salary'] = df['year_0']

    # 5. Years Remaining
    df['years_remaining'] = df['contract_length'].apply(lambda x: max(0, x - 1))
    
    # Sort
    df = df.sort_values('total_contract_value', ascending=False)

    # Clean for Output (Float -> String, 0 -> Empty)
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