"""
Sieve CLI Tool.
This module provides a command-line interface for the Sieve analytics engine.
It allows users to run the data processing pipeline, generate reports, and export
results to CSV files.
"""

import pandas as pd
import numpy as np
import os
import data_processing

class SieveApp:
    """
    Main application class for the Sieve CLI tool.
    
    Attributes:
        lebron_file (str): Path to the LEBRON impact data CSV.
        contracts_file (str): Path to the contracts data CSV.
        df (pd.DataFrame): The merged and processed player DataFrame.
        df_teams (pd.DataFrame): The aggregated team metrics DataFrame.
    """
    
    def __init__(self, lebron_file='LEBRON.csv', contracts_file='basketball_reference_contracts.csv'):
        """
        Initializes the SieveApp with file paths.
        
        Args:
            lebron_file (str): Path to LEBRON data.
            contracts_file (str): Path to contracts data.
        """
        self.lebron_file = lebron_file
        self.contracts_file = contracts_file
        self.df = None
        self.df_teams = None

    def load_data(self):
        """
        Loads and merges data using the shared data_processing module.
        
        This method orchestrates the data loading pipeline:
        1. Load and merge raw CSVs.
        2. Calculate player value metrics.
        3. Calculate team efficiency metrics.
        """
        print("Loading data...")
        try:
            # Load and merge
            self.df = data_processing.load_and_merge_data(self.lebron_file, self.contracts_file)
            
            # Add salary alias for backward compatibility with older analysis scripts if needed
            if 'current_year_salary' in self.df.columns:
                self.df['salary'] = self.df['current_year_salary']
            
            # Calculate metrics
            self.df = data_processing.calculate_player_value_metrics(self.df)
            
            print(f"Successfully loaded {len(self.df)} players.")
            
            # Process team data
            self.process_team_aggregation()
            
        except Exception as e:
            print(f"Error loading data: {e}")

    def process_team_aggregation(self):
        """
        Aggregates player data to calculate team-level metrics.
        
        This wrapper calls the shared calculate_team_metrics function and
        stores the result in self.df_teams.
        """
        if self.df is not None and not self.df.empty:
            self.df_teams = data_processing.calculate_team_metrics(self.df)
            print(f"Aggregated data for {len(self.df_teams)} teams.")

    def analyze_players(self):
        """
        FOR TESTING PURPOSES ONLY
        Performs analysis on player value and prints top lists to the console.
        
        It identifies:
        1. Top 10 Underpaid Players (High Impact, Low Salary).
        2. Top 10 Overpaid Players (Low Impact, High Salary).
        
        It also exports the full analysis to 'sieve_player_analysis.csv'.
        """
        if self.df is None:
            print("No data loaded.")
            return

        print("\n--- Top 10 Underpaid Players (Best Value) ---")
        underpaid = self.df.sort_values('value_gap', ascending=False).head(10)
        print(underpaid[['player_name', 'current_year_salary', 'LEBRON', 'value_gap']].to_string(index=False))

        print("\n--- Top 10 Overpaid Players (Worst Value) ---")
        overpaid = self.df.sort_values('value_gap', ascending=True).head(10)
        print(overpaid[['player_name', 'current_year_salary', 'LEBRON', 'value_gap']].to_string(index=False))
        
        # Export results
        self.df.to_csv('sieve_player_analysis.csv', index=False)
        print("\nFull player analysis saved to 'sieve_player_analysis.csv'")

    def analyze_teams(self):
        """
        Performs analysis on team efficiency and prints rankings to the console.
        
        It ranks teams by their Efficiency Index and exports the results
        to 'sieve_team_efficiency.csv'.
        """
        if self.df_teams is None:
            print("No team data available.")
            return
            
        print("\n--- Team Efficiency Rankings ---")
        ranked_teams = self.df_teams.sort_values('Efficiency_Index', ascending=False)
        
        # Select columns to display
        cols_to_show = ['Abbrev', 'WINS', 'Total_Payroll', 'Efficiency_Index']
        # Filter for columns that actually exist
        display_cols = [c for c in cols_to_show if c in ranked_teams.columns]
        
        print(ranked_teams[display_cols].to_string(index=False))
        
        # Export results
        ranked_teams.to_csv('sieve_team_efficiency.csv', index=False)
        print("\nTeam efficiency analysis saved to 'sieve_team_efficiency.csv'")

    def run(self):
        """
        Executes the full analysis workflow.
        """
        self.load_data()
        self.analyze_players()
        self.analyze_teams()

if __name__ == "__main__":
    app = SieveApp()
    app.run()