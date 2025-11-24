import dash
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from . import data_processing
from . import visualizations
from . import layout

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

print("Loading historical data for Similarity Engine...")
try:
    # Load historical data (cached or fetch new)
    df_history = data_processing.fetch_historical_data()
    
    # Build KNN model with enhanced features
    result = data_processing.build_similarity_model(df_history)
    
    if result and len(result) == 4:
        knn_model, knn_scaler, df_model_data, knn_feature_info = result
    else:
        knn_model, knn_scaler, df_model_data, knn_feature_info = None, None, pd.DataFrame(), None
    
    # Get list of ALL players for dropdown (not just current season)
    if not df_history.empty:
        all_players = df_history['PLAYER_NAME'].unique()
        player_options = [{'label': p, 'value': p} for p in sorted(all_players)]
    else:
        player_options = []
        
    print("Similarity Engine initialized.")
except Exception as e:
    print(f"Error initializing Similarity Engine: {e}")
    import traceback
    traceback.print_exc()
    df_history = pd.DataFrame()
    knn_model = None
    knn_scaler = None
    player_options = []
    df_model_data = pd.DataFrame()
    knn_feature_info = None

# 3. LAYOUT CONSTRUCTION

# Pre-generate static team charts since they don't depend on user filters
# This improves initial load performance.
fig_quadrant = visualizations.create_efficiency_quadrant(df_teams)
fig_grid = visualizations.create_team_grid(df_teams)

# Generate the layout for each tab
player_tab_layout = layout.create_player_tab(df)
team_tab_layout = layout.create_team_tab(df_teams, fig_quadrant, fig_grid)
similarity_tab_layout = layout.create_similarity_tab(player_options)

# Assemble the main application layout
app.layout = layout.create_main_layout()

@app.callback(
    Output('page-content', 'children'),
    Input('view-selector', 'value')
)
def display_content(selected_view):
    """Updates the main page content based on the selected view."""
    if selected_view == 'team':
        return team_tab_layout
    elif selected_view == 'similarity':
        return similarity_tab_layout
    return player_tab_layout


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
     Input('max-salary', 'value'),
     Input('player-search-input', 'value')]
)
def update_dashboard(min_lebron, min_salary, max_salary, search_query):
    """
    Updates all visualizations on the Player Analysis tab based on user filters.
    
    This function is triggered whenever a user adjusts the salary or impact sliders,
    or types in the player search box.
    It filters the global DataFrame and regenerates all charts and tables.

    Args:
        min_lebron (float): Minimum LEBRON impact score.
        min_salary (int): Minimum salary in millions.
        max_salary (int): Maximum salary in millions.
        search_query (str): Player name search query.

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
    
    # Filter the all players table by search query
    if search_query and search_query.strip():
        search_filtered_df = filtered_df[
            filtered_df['player_name'].str.contains(search_query.strip(), case=False, na=False)
        ]
        table_all = visualizations.create_all_players_table(search_filtered_df)
    else:
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
        empty = go.Figure()
        empty.update_layout(template='plotly_dark', paper_bgcolor='#0f1623')
        return empty
    
    radar_data_1 = data_processing.get_team_radar_data(team1_abbr)
    radar_data_2 = data_processing.get_team_radar_data(team2_abbr)
    return visualizations.create_team_radar_chart(radar_data_1, radar_data_2, team1_abbr, team2_abbr)





@app.callback(
    Output('similarity-season-dropdown', 'options'),
    Output('similarity-season-dropdown', 'value'),
    Input('similarity-player-dropdown', 'value')
)
def update_season_dropdown(player_name):
    """Populates the season dropdown based on selected player."""
    if not player_name or df_model_data.empty:
        return [], None
    
    # Get all seasons for this player
    player_seasons = df_model_data[df_model_data['PLAYER_NAME'] == player_name]['SEASON_ID'].unique()
    season_options = [{'label': s, 'value': s} for s in sorted(player_seasons, reverse=True)]
    
    # Default to most recent season
    default_season = sorted(player_seasons, reverse=True)[0] if len(player_seasons) > 0 else None
    
    return season_options, default_season

@app.callback(
    Output('similarity-results-container', 'children'),
    [Input('similarity-player-dropdown', 'value'),
     Input('similarity-season-dropdown', 'value'),
     Input('similarity-exclude-self', 'value')]
)
def update_similarity_results(player_name, season, exclude_self_val):
    """
    Finds and displays similar players based on the selected player and season.
    """
    try:
        if not player_name or not season or knn_model is None:
            return html.Div("Select a player and season to see comparisons.", className="text-center text-muted mt-5")
        
        exclude_self = 'exclude' in (exclude_self_val or [])
        print(f"Searching for similar players to: {player_name} ({season}) (Exclude Self: {exclude_self})")
        
        # Get Target Player Data
        player_rows = df_model_data[(df_model_data['PLAYER_NAME'] == player_name) & (df_model_data['SEASON_ID'] == season)]
        
        if player_rows.empty:
             return html.Div(f"Data for {player_name} in {season} is insufficient for comparison (low games played).", className="text-warning")
             
        target_row = player_rows.iloc[0]
        
        # Get Similar Players with enhanced feature set
        results = data_processing.find_similar_players(
            player_name, season, df_model_data, knn_model, knn_scaler, 
            feature_info=knn_feature_info, exclude_self=exclude_self
        )
        
        if not results:
            return html.Div("No matches found (insufficient data).", className="text-warning")

        # --- Build Cards ---
        cards = []
        
        # 1. Target Player Card
        # Extract stats for target - use actual features from model
        if knn_feature_info and 'features' in knn_feature_info:
            features = knn_feature_info['features']
        else:
            features = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'USG_PCT', 'rTS', 'AST_PCT', '3PA_RATE']
        
        target_stats = {}
        for f in features:
            if f in target_row:
                target_stats[f] = target_row[f]
        
        # Get target position
        target_position = target_row.get('POSITION_GROUP', 'wing').title() if 'POSITION_GROUP' in target_row else 'Wing'
        
        cards.append(layout.create_similarity_card(
            player_name, season, target_row['PLAYER_ID'], target_stats, 
            position=target_position, is_target=True
        ))
        
        # 2. Similar Player Cards
        for res in results:
            cards.append(layout.create_similarity_card(
                res['Player'], res['Season'], res['id'], res['Stats'], 
                position=res.get('Position', 'Wing'),
                match_score=res.get('MatchScore'), distance=res.get('Distance')
            ))
            
        # Layout: 2 Rows of 3 Cards (Total 6)
        return html.Div([
            dbc.Row(cards, className="g-4")
        ])
        
    except Exception as e:
        print("ERROR IN CALLBACK:")
        print(traceback.format_exc())
        return html.Div(f"An error occurred: {str(e)}", className="text-danger")


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
    # use_reloader=False prevents termios errors in some environments
    app.run(debug=True, use_reloader=False)
