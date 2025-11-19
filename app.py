import pandas as pd
import numpy as np
import os


CONFIG = {
    'lebron_file': 'LEBRON.csv',
    'contracts_file': 'basketball_reference_contracts.csv',  # Changed to Basketball-Reference
    'output_file': 'sieve_analysis.csv',
}




class SieveApp:
    """Pipeline to analyze salary by archetype using existing classifications."""
    
    def __init__(self):
        self.df_lebron = None
        self.df_contracts = None
        self.df_merged = None
    
    def load_data(self):
        """Load and validate input files."""
        print("=" * 80)
        print("SIEVE: NBA Archetypes & Salary Analysis")
        print("=" * 80)
        print()
        
        # Load LEBRON data
        if not os.path.exists(CONFIG['lebron_file']):
            print(f"ERROR: {CONFIG['lebron_file']} not found!")
            return False
        
        print(f"[1/2] Loading {CONFIG['lebron_file']}...")
        self.df_lebron = pd.read_csv(CONFIG['lebron_file'])
        print(f"Loaded {len(self.df_lebron)} players")
        
        # Load contract data
        if not os.path.exists(CONFIG['contracts_file']):
            print(f"ERROR: {CONFIG['contracts_file']} not found!")
            return False
        
        print(f"[2/2] Loading {CONFIG['contracts_file']}...")
        self.df_contracts = pd.read_csv(CONFIG['contracts_file'])
        print(f"Loaded {len(self.df_contracts)} contract records")
        
        return True
    
    def process_data(self):
        """Process and merge data."""
        print("\nProcessing data...")
        
        # Ensure player name column exists
        if 'Player' not in self.df_lebron.columns:
            print("ERROR: 'Player' column not found in LEBRON.csv")
            return False
        
        if 'player_name' not in self.df_contracts.columns:
            print("ERROR: 'player_name' column not found in basketball_reference_contracts.csv")
            return False
        
        # Rename for consistency
        self.df_lebron = self.df_lebron.rename(columns={'Player': 'player_name'})
        
        # Convert LEBRON columns to numeric
        for col in ['LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON']:
            if col in self.df_lebron.columns:
                self.df_lebron[col] = pd.to_numeric(self.df_lebron[col], errors='coerce')
        
        # Create combined archetype from existing columns
        print("  Creating combined archetype labels...")
        if 'Offensive Archetype' in self.df_lebron.columns and 'Defensive Role' in self.df_lebron.columns:
            self.df_lebron['archetype'] = (
                self.df_lebron['Offensive Archetype'].fillna('Unknown').astype(str) + 
                ' / ' + 
                self.df_lebron['Defensive Role'].fillna('Unknown').astype(str)
            )
        else:
            print("  WARNING: Archetype columns not found, using generic classification")
            self.df_lebron['archetype'] = 'Unknown'
        
        # Merge with contract data (inner join to only get players with salary data)
        print("  Merging with contract data...")
        self.df_merged = pd.merge(
            self.df_lebron, 
            self.df_contracts,
            on='player_name',
            how='inner'
        )
        
        # Remove duplicates, keep first
        self.df_merged = self.df_merged.drop_duplicates(subset=['player_name'], keep='first')
        
        print(f"  ✓ Merged {len(self.df_merged)} players with salary data")
        
        # Handle NaN in current_year_salary - fill with average_annual_value if available
        if 'current_year_salary' in self.df_merged.columns:
            nan_mask = self.df_merged['current_year_salary'].isna()
            if 'average_annual_value' in self.df_merged.columns:
                self.df_merged.loc[nan_mask, 'current_year_salary'] = self.df_merged.loc[nan_mask, 'average_annual_value']
        
        return True
    
    def analyze(self):
        """Run analysis."""
        print("\n" + "=" * 80)
        print("ARCHETYPE SALARY ANALYSIS")
        print("=" * 80)
        
        # Archetype breakdown
        print("\n=== ARCHETYPES BY SALARY ===\n")
        archetype_stats = self.df_merged.groupby('archetype', observed=True).agg({
            'player_name': 'count',
            'current_year_salary': ['mean', 'median', 'max', 'min'],
            'LEBRON': 'mean',
            'LEBRON WAR': 'mean',
            'O-LEBRON': 'mean',
            'D-LEBRON': 'mean'
        }).round(2)
        
        archetype_stats.columns = ['Count', 'Avg Salary', 'Median Salary', 'Max Salary', 'Min Salary', 
                                   'Avg LEBRON', 'Avg WAR', 'Avg Offense', 'Avg Defense']
        archetype_stats = archetype_stats.sort_values('Avg Salary', ascending=False)
        
        print(archetype_stats.to_string())
        
        # Top earners by archetype
        print("\n\n=== TOP EARNERS BY ARCHETYPE (Top 3 per role) ===\n")
        for archetype in archetype_stats.index:
            if archetype != 'Unknown':
                top = self.df_merged[self.df_merged['archetype'] == archetype].nlargest(3, 'current_year_salary')
                if len(top) > 0:
                    print(f"{archetype}:")
                    for i, (_, row) in enumerate(top.iterrows(), 1):
                        salary = row.get('current_year_salary', 0)
                        lebron = row.get('LEBRON', 0)
                        player_name = str(row['player_name']).strip() if pd.notna(row['player_name']) else 'Unknown'
                        print(f"  {i}. {player_name:25s} | ${salary:12,.0f} | LEBRON: {lebron:6.2f}")
        
        # Correlation
        print("\n\n=== CORRELATIONS ===\n")
        if 'current_year_salary' in self.df_merged.columns:
            corr_cols = ['current_year_salary', 'LEBRON', 'LEBRON WAR', 'O-LEBRON', 'D-LEBRON']
            available = [col for col in corr_cols if col in self.df_merged.columns]
            if len(available) > 1:
                corr_matrix = self.df_merged[available].corr()
                print("Salary correlations:")
                for col in available:
                    if col != 'current_year_salary':
                        val = corr_matrix.loc['current_year_salary', col]
                        print(f"  Salary ↔ {col:20s}: {val:7.4f}")
    
    def export(self):
        """Save analysis."""
        output_path = CONFIG['output_file']
        self.df_merged.to_csv(output_path, index=False)
        
        print("\n" + "=" * 80)
        print("OUTPUT")
        print("=" * 80)
        print(f"\n✓ Saved: {output_path}")
        print(f"  Rows: {len(self.df_merged)}")
        print(f"  Columns: {len(self.df_merged.columns)}")
    
    def run(self):
        """Execute pipeline."""
        if not self.load_data():
            return False
        if not self.process_data():
            return False
        self.analyze()
        self.export()
        
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        
        return self.df_merged

if __name__ == '__main__':
    app = SieveApp()
    df = app.run()
