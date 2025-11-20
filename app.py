import pandas as pd
import numpy as np
import os
import time
from nba_api.stats.endpoints import leaguestandings

CONFIG = {
    'lebron_file': 'LEBRON.csv',
    'contracts_file': 'basketball_reference_contracts.csv',
    'output_file_players': 'sieve_player_analysis.csv',
    'output_file_teams': 'sieve_team_efficiency.csv',
    'standings_cache': 'nba_standings_cache.csv',
    'season': '2024-25'
}

# Maps API Team Names to LEBRON/B-Ref Abbreviations
TEAM_NAME_TO_ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

# Maps standard variations in source files to standard abbreviations
ABBR_NORMALIZATION = {
    'PHO': 'PHX', 'CHO': 'CHA', 'BRK': 'BKN', 'NOH': 'NOP', 'TOT': 'UNK'
}

def fetch_standings(season=CONFIG['season'], cache_file=CONFIG['standings_cache']):
    # Check cache first
    if os.path.exists(cache_file):
        print(f"Loading standings from cache: {cache_file}")
        return pd.read_csv(cache_file)

    print(f"Fetching standings for {season} via NBA API...")
    try:
        # Single API call for all teams
        standings = leaguestandings.LeagueStandings(season=season)
        
        # FIX: Use get_data_frames()[0] instead of get_data_frame()
        dfs = standings.get_data_frames()
        if not dfs:
            print("API returned no data frames.")
            return pd.DataFrame()
            
        df = dfs[0]
        
        # Keep essential columns
        cols = ['TeamID', 'TeamCity', 'TeamName', 'WINS', 'LOSSES', 'WinPCT']
        # Use list comprehension to only select columns that actually exist in the response
        available_cols = [c for c in cols if c in df.columns]
        df_filtered = df[available_cols].copy()
        
        # Create Full Name for mapping
        if 'TeamCity' in df_filtered.columns and 'TeamName' in df_filtered.columns:
            df_filtered['Full_Name'] = df_filtered['TeamCity'] + ' ' + df_filtered['TeamName']
        else:
            print("Warning: TeamCity or TeamName columns missing from API response.")
            # Fallback if needed, though unlikely
            df_filtered['Full_Name'] = ''
        
        # Save to cache
        df_filtered.to_csv(cache_file, index=False)
        time.sleep(0.6) # Rate limit safety
        return df_filtered
    except Exception as e:
        print(f"API Error: {e}")
        # Return empty DF so pipeline doesn't crash, just skips team analysis
        return pd.DataFrame()

class SieveApp:
    def __init__(self):
        self.df_lebron = None
        self.df_contracts = None
        self.df_standings = None
        self.df_merged = None
        self.df_teams = None
    
    def load_data(self):
        print("=" * 80)
        print("SIEVE: NBA Archetypes & Efficiency Analysis")
        print("=" * 80)
        
        if not os.path.exists(CONFIG['lebron_file']) or not os.path.exists(CONFIG['contracts_file']):
            print("Error: Missing input files (LEBRON.csv or contracts file).")
            return False
            
        self.df_lebron = pd.read_csv(CONFIG['lebron_file'])
        self.df_contracts = pd.read_csv(CONFIG['contracts_file'])
        self.df_standings = fetch_standings()
        
        print(f"Loaded {len(self.df_lebron)} LEBRON records and {len(self.df_contracts)} contracts.")
        return True
    
    def process_data(self):
        print("\nProcessing data...")
        
        # 1. Standardize Player Names
        if 'Player' in self.df_lebron.columns:
            self.df_lebron = self.df_lebron.rename(columns={'Player': 'player_name'})
        
        self.df_lebron['player_name'] = self.df_lebron['player_name'].astype(str).str.strip()
        self.df_contracts['player_name'] = self.df_contracts['player_name'].astype(str).str.strip()
        
        # 2. Numeric Conversions for LEBRON stats
        for col in ['LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON']:
            if col in self.df_lebron.columns:
                self.df_lebron[col] = pd.to_numeric(self.df_lebron[col], errors='coerce').fillna(0)
        
        # 3. Create Combined Archetypes (from your original code)
        if 'Offensive Archetype' in self.df_lebron.columns and 'Defensive Role' in self.df_lebron.columns:
            self.df_lebron['archetype'] = (
                self.df_lebron['Offensive Archetype'].fillna('Unknown').astype(str) + 
                ' / ' + 
                self.df_lebron['Defensive Role'].fillna('Unknown').astype(str)
            )
        else:
            self.df_lebron['archetype'] = 'Unknown'

        # 4. Merge Players and Contracts
        self.df_merged = pd.merge(self.df_lebron, self.df_contracts, on='player_name', how='inner')
        
        # 5. Clean Salary Data
        if 'current_year_salary' in self.df_merged.columns:
            # Fill NaN with AAV if available
            if 'average_annual_value' in self.df_merged.columns:
                 self.df_merged['current_year_salary'] = self.df_merged['current_year_salary'].fillna(self.df_merged['average_annual_value'])
            
            # Ensure numeric
            self.df_merged['salary'] = (
                self.df_merged['current_year_salary']
                .astype(str).str.replace(r'[$,]', '', regex=True)
                .pipe(pd.to_numeric, errors='coerce')
                .fillna(0)
            )
            # Update original column for compatibility
            self.df_merged['current_year_salary'] = self.df_merged['salary']
        
        # 6. Deduplicate (handle players with multiple entries, keeping highest salary/main team)
        self.df_merged = self.df_merged.sort_values('salary', ascending=False).drop_duplicates('player_name')
        
        # 7. Normalize Team Abbreviations for Team Analysis
        if 'Team' in self.df_merged.columns:
            self.df_merged['Team'] = self.df_merged['Team'].replace(ABBR_NORMALIZATION)
            
        print(f"Merged dataset contains {len(self.df_merged)} unique players.")
        return True

    def process_team_aggregation(self):
        # Skip if we lack team info
        if self.df_merged is None or 'Team' not in self.df_merged.columns:
            return False

        # Aggregate player stats by team
        team_stats = self.df_merged.groupby('Team').agg({
            'salary': 'sum',
            'LEBRON WAR': 'sum',
            'LEBRON': 'mean',
            'player_name': 'count'
        }).reset_index()
        
        team_stats.rename(columns={
            'salary': 'Total_Payroll',
            'LEBRON WAR': 'Total_WAR',
            'LEBRON': 'Avg_Player_Quality',
            'Team': 'Abbrev'
        }, inplace=True)

        # Merge with API Standings
        if not self.df_standings.empty:
            self.df_standings['Abbrev'] = self.df_standings['Full_Name'].map(TEAM_NAME_TO_ABBR)
            self.df_teams = pd.merge(team_stats, self.df_standings, on='Abbrev', how='left')
        else:
            # If API failed, we still want payroll data, just without Wins
            self.df_teams = team_stats
            
        # Filter invalid rows
        self.df_teams = self.df_teams[self.df_teams['Total_Payroll'] > 0].copy()
        return True

    def analyze_players(self):
        """Original player archetype analysis."""
        print("\n" + "=" * 80)
        print("PLAYER ARCHETYPE ANALYSIS")
        print("=" * 80)
        
        # Archetype Stats
        print("\n--- Archetypes by Salary ---")
        archetype_stats = self.df_merged.groupby('archetype', observed=True).agg({
            'player_name': 'count',
            'salary': ['mean', 'median', 'max', 'min'],
            'LEBRON': 'mean',
            'LEBRON WAR': 'mean'
        }).round(2)
        
        archetype_stats.columns = ['Count', 'Avg Salary', 'Median Salary', 'Max Salary', 'Min Salary', 'Avg LEBRON', 'Avg WAR']
        archetype_stats = archetype_stats.sort_values('Avg Salary', ascending=False)
        print(archetype_stats.to_string())
        
        # Top Earners per Archetype
        print("\n--- Top Earners by Archetype (Top 3) ---")
        for archetype in archetype_stats.index:
            if archetype != 'Unknown':
                top = self.df_merged[self.df_merged['archetype'] == archetype].nlargest(3, 'salary')
                if len(top) > 0:
                    print(f"{archetype}:")
                    for i, (_, row) in enumerate(top.iterrows(), 1):
                        print(f"  {i}. {row['player_name']:25s} | ${row['salary']:12,.0f} | LEBRON: {row['LEBRON']:6.2f}")

        # Correlations
        print("\n--- Salary Correlations ---")
        corr_cols = ['salary', 'LEBRON', 'LEBRON WAR', 'O-LEBRON', 'D-LEBRON']
        available = [col for col in corr_cols if col in self.df_merged.columns]
        if len(available) > 1:
            corr_matrix = self.df_merged[available].corr()
            for col in available:
                if col != 'salary':
                    print(f"  Salary <-> {col:15s}: {corr_matrix.loc['salary', col]:7.4f}")

    def analyze_teams(self):
        """New team efficiency analysis."""
        if self.df_teams is None: return

        print("\n" + "=" * 80)
        print("TEAM EFFICIENCY ANALYSIS")
        print("=" * 80)

        df = self.df_teams.copy()

        # Calculate Basic Metrics
        df['Cost_Per_WAR'] = df['Total_Payroll'] / df['Total_WAR']
        
        if 'WINS' in df.columns:
            df['Cost_Per_Win'] = df['Total_Payroll'] / df['WINS']
            
            # Calculate Z-Scores for Efficiency Index
            # Positive index means high wins relative to payroll (Efficient)
            # Negative index means low wins relative to payroll (Inefficient)
            payroll_z = (df['Total_Payroll'] - df['Total_Payroll'].mean()) / df['Total_Payroll'].std()
            wins_z = (df['WINS'] - df['WINS'].mean()) / df['WINS'].std()
            df['Efficiency_Index'] = wins_z - payroll_z
        else:
            print("Warning: 'WINS' data not found (API failure?). Skipping Win-based metrics.")
            df['Cost_Per_Win'] = 0
            df['Efficiency_Index'] = 0

        # Format for display
        df['Payroll_M'] = df['Total_Payroll'] / 1_000_000
        df['CPW_M'] = df['Cost_Per_Win'] / 1_000_000
        
        # Sort by Efficiency Index
        df = df.sort_values('Efficiency_Index', ascending=False)
        
        # Display
        cols = ['Abbrev', 'WINS', 'Payroll_M', 'Total_WAR', 'CPW_M', 'Efficiency_Index']
        # Only show columns that actually exist
        final_cols = [c for c in cols if c in df.columns]
        display_df = df[final_cols].round(2)
        
        print("\n--- Teams Ranked by Salary Efficiency (Index) ---")
        print(display_df.to_string(index=False))
        
        if 'CPW_M' in display_df.columns:
            print("\n--- Best Value (Lowest Cost Per Win) ---")
            print(display_df.sort_values('CPW_M').head(3).to_string(index=False))

        # Export Team Data
        df.to_csv(CONFIG['output_file_teams'], index=False)
        print(f"\nSaved team analysis to {CONFIG['output_file_teams']}")

    def run(self):
        if self.load_data() and self.process_data():
            # Player Analysis
            self.analyze_players()
            self.df_merged.to_csv(CONFIG['output_file_players'], index=False)
            
            # Team Analysis
            if self.process_team_aggregation():
                self.analyze_teams()

if __name__ == '__main__':
    app = SieveApp()
    app.run()