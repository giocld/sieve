import dash
from dash import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

import data_processing
import visualizations
import layout

# 1. APP INITIALIZATION

# Initialize the Dash app with Bootstrap stylesheets.
# We suppress callback exceptions to allow for content loading.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "Sieve | NBA Analytics"

# Expose the Flask server for deployment (e.g., Gunicorn)
server = app.server

# 2. DATA LOADING & PROCESSING

print("Loading data...")
try:
    # Load and merge player data from CSVs
    df = data_processing.load_and_merge_data()
    
    # Calculate player-level value metrics (Value Gap)
    df = data_processing.calculate_player_value_metrics(df)
    
    # Calculate team-level efficiency metrics
    df_teams = data_processing.calculate_team_metrics(df)
    
    # Fetch and add official NBA team logos
    df_teams = data_processing.add_team_logos(df_teams)
    
    print("Data loaded successfully.")
    
except Exception as e:
    print(f"Error loading data: {e}")
    # Create empty DataFrames to prevent app crash on load failure
    df = pd.DataFrame()
    df_teams = pd.DataFrame()

# 3. LAYOUT CONSTRUCTION

# Pre-generate static team charts since they don't depend on user filters
# This improves initial load performance.
fig_quadrant = visualizations.create_efficiency_quadrant(df_teams)
fig_grid = visualizations.create_team_grid(df_teams)

# Generate the layout for each tab
player_tab_layout = layout.create_player_tab(df)
team_tab_layout = layout.create_team_tab(df_teams, fig_quadrant, fig_grid)

# Assemble the main application layout
app.layout = layout.create_main_layout(player_tab_layout, team_tab_layout)


# 4.  CALLBACKS
@app.callback(
    Output('min-salary', 'value'),
    Input('max-salary', 'value'),
    State('min-salary', 'value')
)
def enforce_salary_constraints(max_val, current_min):
    """
    Ensures the minimum salary slider value never exceeds the maximum slider value.
    
    Args:
        max_val (int): The current value of the max salary slider.
        current_min (int): The current value of the min salary slider.
        
    Returns:
        int: The adjusted minimum salary value.
    """
    if current_min > max_val:
        return max_val
    return current_min

@app.callback(
    [Output('chart-salary-impact', 'figure'),
     Output('chart-underpaid', 'figure'),
     Output('chart-off-def', 'figure'),
     Output('chart-overpaid', 'figure'),
     Output('table-underpaid', 'children'),
     Output('table-overpaid', 'children'),
     Output('table-all-players', 'children')],
    [Input('min-lebron', 'value'),
     Input('min-salary', 'value'),
     Input('max-salary', 'value')]
)
def update_dashboard(min_lebron, min_salary, max_salary):
    """
    Updates all visualizations on the Player Analysis tab based on user filters.
    
    This function is triggered whenever a user adjusts the salary or impact sliders.
    It filters the global DataFrame and regenerates all charts and tables.

    Args:
        min_lebron (float): Minimum LEBRON impact score.
        min_salary (int): Minimum salary in millions.
        max_salary (int): Maximum salary in millions.

    Returns:
        tuple: A tuple containing 4 Plotly figures and 3 HTML table components.
    """
    # Filter the DataFrame based on slider inputs
    # We convert millions back to raw dollars for comparison
    filtered_df = df[
        (df['LEBRON'] >= min_lebron) & 
        (df['current_year_salary'] >= min_salary * 1_000_000) & 
        (df['current_year_salary'] <= max_salary * 1_000_000)
    ].copy()
    
    # Recalculate value metrics for the filtered subset
    # This ensures percentiles are relative to the currently selected pool of players
    filtered_df = data_processing.calculate_player_value_metrics(filtered_df)
    
    # Generate updated visualizations
    fig_scatter = visualizations.create_salary_impact_scatter(filtered_df)
    fig_under = visualizations.create_underpaid_bar(filtered_df)
    fig_age = visualizations.create_age_impact_scatter(filtered_df)
    fig_over = visualizations.create_overpaid_bar(filtered_df)
    
    # Generate updated tables
    table_under = visualizations.create_player_table(filtered_df, 'underpaid')
    table_over = visualizations.create_player_table(filtered_df, 'overpaid')
    table_all = visualizations.create_all_players_table(filtered_df)
    
    return fig_scatter, fig_under, fig_age, fig_over, table_under, table_over, table_all




@app.callback(
    Output('chart-team-radar', 'figure'),
    [Input('team-radar-dropdown-1', 'value'),
     Input('team-radar-dropdown-2', 'value')]
)
def update_team_radar(team1_abbr, team2_abbr):
    """Updates the Team Comparison Radar Chart based on selected teams."""
    if not team1_abbr or not team2_abbr:
        from plotly import graph_objects as go
        empty = go.Figure()
        empty.update_layout(template='plotly_dark', paper_bgcolor='#0f1623')
        return empty
    
    radar_data_1 = data_processing.get_team_radar_data(team1_abbr)
    radar_data_2 = data_processing.get_team_radar_data(team2_abbr)
    return visualizations.create_team_radar_chart(radar_data_1, radar_data_2, team1_abbr, team2_abbr)



# 5. ENTRY POINT

if __name__ == '__main__':
    print("======================================================================")
    print("SIEVE Dashboard Starting...")
    print("======================================================================")
    print(f"Loaded {len(df)} players and {len(df_teams)} teams")
    print("Open browser to: http://localhost:8050")
    print("Press CTRL+C to stop")
    print("")
    # Run the application in debug mode for development
    app.run(debug=True)