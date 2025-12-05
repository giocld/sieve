"""
Data Scraper for Sieve NBA Analytics.

This module provides methods to get player data from external sources:

Contract Data (Basketball Reference):
1. parse_bbref_csv() - Parse a manually downloaded CSV (recommended)
2. scrape_bball_ref() - Web scrape using Selenium (fallback)

LEBRON Data (BBall Index):
3. scrape_lebron() - Scrape from https://www.bball-index.com/lebron-application/
4. parse_lebron_csv() - Parse manually downloaded CSV (fallback)

Data is saved to the unified SQLite database via cache_manager.
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

# Import unified cache manager
from src.cache_manager import cache
from src.config import CURRENT_SEASON


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
    
    # Save to database (primary storage)
    cache.save_contracts(df_final, season=CURRENT_SEASON)
    
    # Also save CSV as backup (can be removed after full migration)
    df_final.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(df_final)} players to database and {output_path}")
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
    
    # Save to database (primary storage)
    cache.save_contracts(output_df, season=CURRENT_SEASON)
    
    # Also save CSV as backup (can be removed after full migration)
    output_df.to_csv('data/basketball_reference_contracts.csv', index=False)
    
    return output_df


# =============================================================================
# LEBRON DATA SCRAPING (bball-index.com)
# =============================================================================

def scrape_lebron(season='2025-26', save_csv=True, headless=True, show_all=True, max_retries=3):
    """
    Scrape LEBRON player impact data from BBall Index with retry logic.
    
    Source: https://www.bball-index.com/lebron-application/
    
    LEBRON (Luck-adjusted player Estimate using a Box prior Regularized ON-off)
    is a comprehensive player impact metric that combines box score and on/off data.
    
    Args:
        season (str): NBA season to scrape (e.g., '2024-25' or '2025-26')
        save_csv (bool): Whether to also save a CSV backup
        headless (bool): Run browser in headless mode (set False for debugging)
        show_all (bool): Try to show all entries (not just first page)
        max_retries (int): Number of retry attempts (default: 3)
        
    Returns:
        pd.DataFrame: LEBRON data for all players, or None if scraping failed
    """
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import Select
    
    print("=" * 70)
    print("BBall Index LEBRON Scraper")
    print("Target: https://www.bball-index.com/lebron-application/")
    print(f"Season: {season}")
    print(f"Max retries: {max_retries}")
    print("=" * 70)
    
    url = "https://www.bball-index.com/lebron-application/"
    
    # Extract year from season string (e.g., "2025-26" -> 2026)
    try:
        target_year = int(season[:4]) + 1
    except:
        target_year = 2026
    
    print(f"Target Year for Slider: {target_year}")
    
    for attempt in range(1, max_retries + 1):
        print(f"\n[Attempt {attempt}/{max_retries}]")
        all_data = []
        driver = None
        
        try:
            # Setup Chrome with robust options
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1200")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(120)
            
            print(f"Loading: {url}")
            driver.get(url)
            
            # Wait for initial load with exponential backoff
            wait_time = 8 + (attempt - 1) * 4
            print(f"Waiting {wait_time}s for page to load...")
            time.sleep(wait_time)
            
            wait = WebDriverWait(driver, 60 + attempt * 15)
            
            # Find and switch to the Shiny iframe
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            print(f"Found {len(iframes)} iframes")
            
            shiny_frame = None
            for iframe in iframes:
                try:
                    src = iframe.get_attribute('src') or ''
                    if 'shiny' in src.lower() or 'bball-index' in src.lower() or 'rstudio' in src.lower():
                        shiny_frame = iframe
                        print(f"Found Shiny iframe: {src[:80]}...")
                        break
                except:
                    continue
            
            if shiny_frame:
                driver.switch_to.frame(shiny_frame)
                print("Switched to Shiny iframe")
                time.sleep(5 + attempt)
            else:
                print("No Shiny iframe found, trying main content...")
            
            # ---------------------------------------------------------
            # 1. SET SEASON (Slider or Dropdown)
            # ---------------------------------------------------------
            print(f"Attempting to set season to {season} (Year: {target_year})...")
            
            # Try to find ion-range-slider (common in Shiny)
            slider_found = False
            try:
                # Look for the slider input
                sliders = driver.find_elements(By.CLASS_NAME, "js-range-slider")
                if sliders:
                    print(f"Found {len(sliders)} range sliders. Attempting to update via JavaScript...")
                    for i, slider in enumerate(sliders):
                        # Use jQuery/ionRangeSlider API to update
                        # We want to set BOTH 'from' and 'to' to the target year to filter for just that season
                        js_script = f"""
                        var slider = $(arguments[0]).data("ionRangeSlider");
                        if (slider) {{
                            slider.update({{
                                from: {target_year},
                                to: {target_year}
                            }});
                            return true;
                        }}
                        return false;
                        """
                        result = driver.execute_script(js_script, slider)
                        if result:
                            print(f"Successfully updated slider {i} to {target_year}")
                            slider_found = True
                            time.sleep(2)
            except Exception as e:
                print(f"Error updating slider: {e}")

            # Fallback: Try dropdowns if slider didn't work or wasn't found
            if not slider_found:
                print("Slider update skipped or failed. Checking for dropdowns...")
                # ... (Existing dropdown logic could go here, but slider is primary for this tool)
            
            # ---------------------------------------------------------
            # 2. CLICK "RUN QUERY"
            # ---------------------------------------------------------
            print("Looking for 'Run Query' button...")
            try:
                # Try multiple selectors for the button
                run_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'Run Query')]")
                if not run_btns:
                    run_btns = driver.find_elements(By.XPATH, "//a[contains(text(), 'Run Query')]")
                if not run_btns:
                    run_btns = driver.find_elements(By.ID, "run_query") # Common ID guess
                
                if run_btns:
                    print("Clicking 'Run Query' button...")
                    # Scroll to button
                    driver.execute_script("arguments[0].scrollIntoView(true);", run_btns[0])
                    time.sleep(1)
                    run_btns[0].click()
                    print("Clicked 'Run Query'. Waiting for results...")
                    time.sleep(8) # Wait for query to run
                else:
                    print("Warning: 'Run Query' button not found.")
            except Exception as e:
                print(f"Error clicking 'Run Query': {e}")

            # ---------------------------------------------------------
            # 3. SHOW ALL ENTRIES
            # ---------------------------------------------------------
            if show_all:
                print("Trying to show all entries...")
                show_all_selectors = [
                    "//select[contains(@name, 'length')]",
                    "//select[contains(@class, 'form-control')]",
                    "//*[contains(@class, 'dataTables_length')]//select",
                ]
                
                for selector in show_all_selectors:
                    try:
                        select_elem = driver.find_elements(By.XPATH, selector)
                        if select_elem:
                            # Try to select "All" or the highest number
                            options = select_elem[0].find_elements(By.TAG_NAME, 'option')
                            for opt in reversed(options):  # Start from last (usually "All" or highest)
                                try:
                                    text = opt.text.lower()
                                    if 'all' in text or text.isdigit():
                                        opt.click()
                                        print(f"Selected: Show {opt.text}")
                                        time.sleep(5) # Wait for table redraw
                                        break
                                except:
                                    continue
                            break
                    except:
                        continue
            
            # Wait for data to load
            time.sleep(5)
            
            # ---------------------------------------------------------
            # 4. SCRAPE TABLE
            # ---------------------------------------------------------
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Find DataTables with multiple selectors
            tables = soup.select('table.dataTable, table.display, #DataTables_Table_0, table.table, table')
            print(f"Found {len(tables)} tables")
            
            # Find the best table with player data
            best_table = None
            max_rows = 0
            
            for table in tables:
                rows = table.find_all('tr')
                text = table.get_text().lower()
                # Look for LEBRON-specific indicators
                if len(rows) > max_rows and ('lebron' in text or 'player' in text or 'war' in text):
                    max_rows = len(rows)
                    best_table = table
            
            if best_table and max_rows > 5:
                print(f"Processing table with {max_rows} rows")
                
                # Extract headers
                headers = []
                thead = best_table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                
                if not headers:
                    first_row = best_table.find('tr')
                    if first_row:
                        headers = [c.get_text(strip=True) for c in first_row.find_all(['th', 'td'])]
                
                print(f"Headers ({len(headers)}): {headers}")
                
                # Extract data rows
                tbody = best_table.find('tbody')
                rows = tbody.find_all('tr') if tbody else best_table.find_all('tr')[1:]
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 5:
                        row_data = {}
                        for i, cell in enumerate(cells):
                            col_name = headers[i] if i < len(headers) else f'col_{i}'
                            row_data[col_name] = cell.get_text(strip=True)
                        
                        first_val = str(list(row_data.values())[0]).lower().strip()
                        if first_val in ['player', 'rank', '#', '', 'name', 'number']:
                            continue
                        
                        if any(c.replace('.', '').replace('-', '').isdigit() 
                               for c in list(row_data.values())[-4:]):
                            all_data.append(row_data)
                
                print(f"Extracted {len(all_data)} player rows")
            else:
                print("Could not find suitable data table")
            
            driver.switch_to.default_content()
                
        except Exception as e:
            print(f"Error on attempt {attempt}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if driver:
                driver.quit()
        
        # If we got data, process and return
        if all_data:
            break
        
        # Exponential backoff before retry
        if attempt < max_retries:
            backoff = 5 * (2 ** (attempt - 1))  # 5s, 10s, 20s
            print(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
    
    # Check if scraping was successful
    if not all_data:
        print("\n" + "=" * 70)
        print("AUTOMATED SCRAPING UNSUCCESSFUL")
        print(f"Failed after {max_retries} attempts")
        print("=" * 70)
        print("\nThe BBall Index app requires manual interaction.")
        print("\nManual download instructions:")
        print("  1. Visit: https://www.bball-index.com/lebron-application/")
        print("  2. Select season (e.g., 2025-26) from dropdown")
        print("  3. Set 'Show entries' to 'All' or highest number")
        print("  4. Select all data in the table (Ctrl+A on table)")
        print("  5. Paste into a text file")
        print("  6. Save as: data/LEBRON_raw.csv")
        print("  7. Run: python -m src.scraper --lebron-csv data/LEBRON_raw.csv --season {season}")
        print("=" * 70)
        return None
    
    # Process extracted data
    df = pd.DataFrame(all_data)
    print(f"Raw columns: {list(df.columns)}")
    
    # Standardize column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'player' in col_lower or col_lower == 'name':
            col_map[col] = 'Player'
        elif col_lower in ['team', 'team(s)', 'tm', 'teams']:
            col_map[col] = 'Team(s)'
        elif col_lower == 'age':
            col_map[col] = 'Age'
        elif col_lower in ['min', 'minutes']:
            col_map[col] = 'Minutes'
        elif col_lower == 'lebron war' or col_lower == 'war':
            col_map[col] = 'LEBRON WAR'
        elif col_lower == 'lebron':
            col_map[col] = 'LEBRON'
        elif col_lower == 'o-lebron':
            col_map[col] = 'O-LEBRON'
        elif col_lower == 'd-lebron':
            col_map[col] = 'D-LEBRON'
        elif 'rotation' in col_lower or 'role' in col_lower:
            col_map[col] = 'Rotation Role'
        elif 'offensive' in col_lower or 'off' in col_lower:
            col_map[col] = 'Offensive Archetype'
        elif 'defensive' in col_lower or 'def' in col_lower:
            col_map[col] = 'Defensive Role'
    
    df = df.rename(columns=col_map)
    df['Season'] = season
    
    # Convert numeric columns
    for col in ['Age', 'Minutes', 'LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with no player name
    if 'Player' in df.columns:
        df = df[df['Player'].notna() & (df['Player'] != '')]
    
    print(f"\nProcessed {len(df)} players")
    if 'LEBRON' in df.columns:
        print("\nTop 10 by LEBRON:")
        print(df.nlargest(10, 'LEBRON')[['Player', 'Team(s)', 'LEBRON', 'LEBRON WAR']].to_string(index=False))
    
    # Save to database
    cache.save_lebron_metrics(df, season=season)
    
    if save_csv:
        csv_path = f'data/LEBRON_{season.replace("-", "_")}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
    
    return df


def parse_lebron_csv(csv_path='data/LEBRON.csv', season=None):
    """
    Parse a manually downloaded/copy-pasted LEBRON data from BBall Index.
    
    Handles multiple formats:
    - Standard CSV (comma-separated)
    - Tab-separated (from copy-paste)
    - BBall Index specific format with merged "Number Player" header
    
    How to get the data:
    1. Go to https://www.bball-index.com/lebron-application/
    2. Select the desired season (e.g., 2025-26)
    3. Set "Show entries" to "All" or maximum
    4. Select all table data and copy (Ctrl+A, Ctrl+C on table)
    5. Paste into a text file and save as .csv
    6. Run: python -m src.scraper --lebron-csv <path>
    
    Args:
        csv_path (str): Path to the LEBRON CSV/TSV file
        season (str): Override season value (auto-detected from data if not provided)
        
    Returns:
        pd.DataFrame: Processed LEBRON data
    """
    print("=" * 70)
    print("BBall Index LEBRON CSV Parser")
    print(f"Input: {csv_path}")
    print("=" * 70)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        print("\nTo get LEBRON data:")
        print("1. Go to https://www.bball-index.com/lebron-application/")
        print("2. Select season, set 'Show entries' to All")
        print("3. Copy table data and paste into text file")
        print(f"4. Save as {csv_path}")
        return None
    
    # Read raw file to detect format
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_lines = [f.readline() for _ in range(3)]
    
    # Detect delimiter
    tab_count = sum(line.count('\t') for line in first_lines)
    comma_count = sum(line.count(',') for line in first_lines)
    
    if tab_count > comma_count:
        print("Detected: Tab-separated format (BBall Index copy-paste)")
        delimiter = '\t'
    else:
        print("Detected: Comma-separated format")
        delimiter = ','
    
    # Check for BBall Index specific format (merged "Number Player" header)
    header_line = first_lines[0]
    if 'Number' in header_line and 'Player' in header_line and delimiter == '\t':
        print("Detected: BBall Index raw copy-paste format")
        
        # The data has: Number, (empty), Player, Age, Team, Minutes, Role, Off, Def, WAR, LEBRON, O-LEB, D-LEB
        columns = ['_num', '_empty', 'Player', 'Age', 'Team(s)', 'Minutes', 
                   'Rotation Role', 'Offensive Archetype', 'Defensive Role', 
                   'LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON']
        
        df = pd.read_csv(csv_path, sep='\t', skiprows=1, names=columns, 
                         on_bad_lines='skip', encoding='utf-8')
        
        # Drop helper columns
        df = df.drop(columns=['_num', '_empty'], errors='ignore')
    else:
        # Standard format
        df = pd.read_csv(csv_path, sep=delimiter, on_bad_lines='skip', encoding='utf-8')
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Standardize column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['player', 'name']:
            col_map[col] = 'Player'
        elif col_lower in ['team', 'team(s)', 'tm', 'teams']:
            col_map[col] = 'Team(s)'
        elif col_lower == 'age':
            col_map[col] = 'Age'
        elif col_lower in ['min', 'minutes']:
            col_map[col] = 'Minutes'
        elif col_lower in ['lebron war', 'war']:
            col_map[col] = 'LEBRON WAR'
        elif col_lower == 'lebron':
            col_map[col] = 'LEBRON'
        elif col_lower == 'o-lebron':
            col_map[col] = 'O-LEBRON'
        elif col_lower == 'd-lebron':
            col_map[col] = 'D-LEBRON'
        elif 'rotation' in col_lower:
            col_map[col] = 'Rotation Role'
        elif 'offensive' in col_lower:
            col_map[col] = 'Offensive Archetype'
        elif 'defensive' in col_lower:
            col_map[col] = 'Defensive Role'
    
    df = df.rename(columns=col_map)
    
    # Detect season from data or use provided value
    if season:
        detected_season = season
    elif 'Season' in df.columns:
        detected_season = df['Season'].mode().iloc[0] if not df['Season'].mode().empty else CURRENT_SEASON
    else:
        detected_season = CURRENT_SEASON
    
    print(f"Season: {detected_season}")
    
    # Ensure Season column exists
    if 'Season' not in df.columns:
        df['Season'] = detected_season
    
    # Convert numeric columns
    numeric_cols = ['LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON', 'Minutes', 'Age']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove empty/invalid rows
    df = df.dropna(how='all')
    
    if 'Player' in df.columns:
        df = df[df['Player'].notna()]
        df = df[df['Player'].astype(str).str.strip() != '']
        df = df[~df['Player'].str.lower().isin(['player', 'name', 'number'])]
    
    # Reorder columns for consistency
    desired_order = ['Season', 'Player', 'Age', 'Team(s)', 'Minutes', 'Rotation Role',
                     'Offensive Archetype', 'Defensive Role', 'LEBRON WAR', 'LEBRON', 
                     'O-LEBRON', 'D-LEBRON']
    existing_cols = [c for c in desired_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in desired_order]
    df = df[existing_cols + other_cols]
    
    print(f"Processed {len(df)} valid player records")
    
    # Show sample
    if 'Player' in df.columns and 'LEBRON' in df.columns:
        print("\nTop 10 by LEBRON:")
        top_cols = ['Player', 'Team(s)', 'LEBRON', 'LEBRON WAR', 'O-LEBRON', 'D-LEBRON']
        available_cols = [c for c in top_cols if c in df.columns]
        print(df.nlargest(10, 'LEBRON')[available_cols].to_string(index=False))
    
    # Save to database
    cache.save_lebron_metrics(df, season=detected_season)
    print(f"\nSaved to database (season: {detected_season})")
    
    return df


if __name__ == "__main__":
    import sys
    
    if '--csv' in sys.argv:
        # Use CSV parser for contracts (preferred)
        csv_path = 'data/bbref_contracts_raw.csv'
        if len(sys.argv) > 2 and not sys.argv[-1].startswith('-'):
            csv_path = sys.argv[-1]
        parse_bbref_csv(csv_path)
    
    elif '--lebron' in sys.argv:
        # Scrape LEBRON data from bball-index
        season = CURRENT_SEASON
        for i, arg in enumerate(sys.argv):
            if arg == '--season' and i + 1 < len(sys.argv):
                season = sys.argv[i + 1]
        scrape_lebron(season=season)
    
    elif '--lebron-csv' in sys.argv:
        # Parse LEBRON CSV
        csv_path = 'data/LEBRON.csv'
        season = None
        for i, arg in enumerate(sys.argv):
            if arg == '--lebron-csv' and i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                csv_path = sys.argv[i + 1]
            if arg == '--season' and i + 1 < len(sys.argv):
                season = sys.argv[i + 1]
        parse_lebron_csv(csv_path, season=season)
    
    elif '--help' in sys.argv or '-h' in sys.argv:
        print("Sieve Data Scraper")
        print("-" * 50)
        print("\nUsage: python -m src.scraper [options]")
        print("\nContract Data (Basketball Reference):")
        print("  --csv [path]              Parse BBRef contracts CSV")
        print("  (no args)                 Scrape BBRef with Selenium")
        print("\nLEBRON Data (BBall Index):")
        print("  --lebron [--season YYYY-YY]   Scrape from bball-index.com")
        print("  --lebron-csv [path] [--season YYYY-YY]  Parse LEBRON CSV")
        print("\nExamples:")
        print("  python -m src.scraper --csv")
        print("  python -m src.scraper --lebron --season 2025-26")
        print("  python -m src.scraper --lebron-csv data/LEBRON.csv")
    
    else:
        # Default: Use Selenium scraper for contracts
        scrape_bball_ref()
