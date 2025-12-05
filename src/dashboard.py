import dash
from dash import Input, Output, State, html, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from . import data_processing
from . import visualizations
from . import layout
from .config import CURRENT_SEASON
from .cache_manager import cache

# 1. APP INITIALIZATION

# Initialize the Dash app with Bootstrap stylesheets.
# We suppress callback exceptions to allow for content loading.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "Sieve | NBA Analytics"

# Expose the Flask server for deployment (e.g., Gunicorn)
server = app.server

# 2. SEASON-AWARE DATA MANAGEMENT

# Cache for season-specific data (loaded on demand)
_season_data_cache = {}

def load_season_data(season):
    """
    Load data for a specific season.
    Caches results to avoid reloading.
    
    Args:
        season (str): NBA season string (e.g., '2024-25')
        
    Returns:
        tuple: (df_players, df_teams) DataFrames
    """
    if season in _season_data_cache:
        return _season_data_cache[season]
    
    print(f"Loading data for season {season}...")
    
    try:
        # Determine LEBRON file
        if season == '2024-25':
            lebron_file = 'data/LEBRON.csv'
        else:
            lebron_file = f'data/LEBRON_{season.replace("-", "_")}.csv'
        
        # Load and merge player data
        df_players = data_processing.load_and_merge_data(
            lebron_file=lebron_file,
            season=season,
            from_db=True
        )
        
        # Calculate player-level value metrics
        df_players = data_processing.calculate_player_value_metrics(df_players, season=season)
        
        # Calculate team-level efficiency metrics
        df_teams = data_processing.calculate_team_metrics(df_players, season=season)
        
        # Add team logos
        df_teams = data_processing.add_team_logos(df_teams)
        
        # Cache the results
        _season_data_cache[season] = (df_players, df_teams)
        print(f"Loaded {len(df_players)} players, {len(df_teams)} teams for {season}")
        
        return df_players, df_teams
        
    except Exception as e:
        print(f"Error loading data for {season}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()


def get_current_data(season=None):
    """Get data for the specified season (or current season)."""
    season = season or CURRENT_SEASON
    return load_season_data(season)


# 3. INITIAL DATA LOADING (default season)

print("Loading initial data...")
df, df_teams = load_season_data(CURRENT_SEASON)

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

# Generate lineup team options
print("Loading lineup team options...")
try:
    lineup_team_options = data_processing.get_lineup_teams()
except Exception as e:
    print(f"Error loading lineup teams: {e}")
    lineup_team_options = []

# Generate the layout for each tab
landing_tab_layout = layout.create_landing_tab(df, df_teams)
player_tab_layout = layout.create_player_tab(df)
team_tab_layout = layout.create_team_tab(df_teams, fig_quadrant, fig_grid)
lineup_tab_layout = layout.create_lineup_tab(lineup_team_options)
similarity_tab_layout = layout.create_similarity_tab(player_options)

# Assemble the main application layout
app.layout = layout.create_main_layout()

# Callback to handle card navigation requests
@app.callback(
    Output('nav-request', 'data'),
    [Input('card-nav-player', 'n_clicks'),
     Input('card-nav-team', 'n_clicks'),
     Input('card-nav-lineup', 'n_clicks'),
     Input('card-nav-similarity', 'n_clicks')],
    prevent_initial_call=True
)
def handle_card_navigation(player, team, lineup, similarity):
    """Handles clicks on quick access cards."""
    from dash import ctx
    card_to_nav = {
        'card-nav-player': 'nav-player',
        'card-nav-team': 'nav-team',
        'card-nav-lineup': 'nav-lineup',
        'card-nav-similarity': 'nav-similarity'
    }
    if ctx.triggered_id in card_to_nav:
        return card_to_nav[ctx.triggered_id]
    return None

@app.callback(
    [Output('page-content', 'children'),
     Output('view-selector', 'data'),
     Output('nav-home', 'active'),
     Output('nav-player', 'active'),
     Output('nav-team', 'active'),
     Output('nav-lineup', 'active'),
     Output('nav-similarity', 'active'),
     Output('season-data-store', 'data')],
    [Input('nav-brand', 'n_clicks'),
     Input('nav-home', 'n_clicks'),
     Input('nav-player', 'n_clicks'),
     Input('nav-team', 'n_clicks'),
     Input('nav-lineup', 'n_clicks'),
     Input('nav-similarity', 'n_clicks'),
     Input('nav-request', 'data'),
     Input('season-selector', 'value')],
    [State('view-selector', 'data')],
    prevent_initial_call=False
)
def display_content(brand_clicks, home_clicks, player_clicks, team_clicks, lineup_clicks, similarity_clicks,
                   nav_request, selected_season, current_view):
    """Updates the main page content based on navigation clicks and season selection."""
    from dash import ctx
    
    # Use selected season or default
    season = selected_season or CURRENT_SEASON
    
    # Load data for the selected season
    df_season, df_teams_season = get_current_data(season)
    
    # Check if this is initial load (no actual clicks yet)
    all_clicks = [brand_clicks, home_clicks, player_clicks, team_clicks, lineup_clicks, similarity_clicks]
    is_initial_load = all(c is None or c == 0 for c in all_clicks) and nav_request is None
    
    # If season changed, don't switch tabs - just reload current view
    season_changed = ctx.triggered_id == 'season-selector'
    
    # Determine which nav was clicked
    if is_initial_load or not ctx.triggered_id:
        triggered_id = 'nav-home'
    elif season_changed:
        # Keep current view when season changes
        triggered_id = f'nav-{current_view}' if current_view else 'nav-home'
    elif ctx.triggered_id == 'nav-request' and nav_request:
        # Navigation request from card click
        triggered_id = nav_request
    elif ctx.triggered_id == 'nav-brand':
        # Brand click goes to home
        triggered_id = 'nav-home'
    else:
        triggered_id = ctx.triggered_id
    
    # Generate tab layouts with season-specific data
    fig_quadrant = visualizations.create_efficiency_quadrant(df_teams_season)
    fig_grid = visualizations.create_team_grid(df_teams_season)
    
    # Map nav IDs to views and content (regenerate with current season data)
    nav_map = {
        'nav-home': ('home', layout.create_landing_tab(df_season, df_teams_season)),
        'nav-player': ('player', layout.create_player_tab(df_season)),
        'nav-team': ('team', layout.create_team_tab(df_teams_season, fig_quadrant, fig_grid)),
        'nav-lineup': ('lineup', lineup_tab_layout),  # Lineups are fetched dynamically
        'nav-similarity': ('similarity', similarity_tab_layout)  # Uses historical data
    }
    
    # Get selected view and content
    selected_view, content = nav_map.get(triggered_id, ('home', layout.create_landing_tab(df_season, df_teams_season)))
    
    # Set active states for nav pills
    nav_active = [triggered_id == nav_id for nav_id in ['nav-home', 'nav-player', 'nav-team', 'nav-lineup', 'nav-similarity']]
    
    # Store season info
    season_store = {'season': season}
    
    return [content, selected_view] + nav_active + [season_store]


# 4.  CALLBACKS
@app.callback(
    [Output('min-salary', 'value'),
     Output('max-salary', 'value')],
    [Input('min-salary', 'value'),
     Input('max-salary', 'value')],
    prevent_initial_call=True
)
def enforce_salary_constraints(min_val, max_val):
    """
    Ensures salary slider constraints are maintained.
    
    Rules:
    - If min > max: set min = max - 1 (or 0 if max < 1)
    - If max < min: set max = min + 1
    
    Args:
        min_val (int): Current value of min salary slider.
        max_val (int): Current value of max salary slider.
        
    Returns:
        tuple: (adjusted_min, adjusted_max)
    """
    from dash import ctx
    
    # Which slider was changed?
    triggered = ctx.triggered_id
    
    if triggered == 'min-salary':
        # User changed min - if min > max, push max up
        if min_val > max_val:
            return min_val, min_val + 1
        return min_val, max_val
    elif triggered == 'max-salary':
        # User changed max - if max < min, push min down
        if max_val < min_val:
            new_min = max(0, max_val - 1)
            return new_min, max_val
        return min_val, max_val
    
    # Fallback
    return min_val, max_val

@app.callback(
    [Output('chart-salary-impact', 'figure'),
     Output('chart-underpaid', 'figure'),
     Output('chart-beeswarm', 'figure'),
     Output('chart-overpaid', 'figure'),
     Output('table-underpaid', 'children'),
     Output('table-overpaid', 'children'),
     Output('table-all-players', 'children')],
    [Input('min-lebron', 'value'),
     Input('min-salary', 'value'),
     Input('max-salary', 'value'),
     Input('player-search-input', 'value'),
     Input('season-data-store', 'data')]
)
def update_dashboard(min_lebron, min_salary, max_salary, search_query, season_data):
    """
    Updates all visualizations on the Player Analysis tab based on user filters.
    
    This function is triggered whenever a user adjusts the salary or impact sliders,
    types in the player search box, or changes the season.

    Args:
        min_lebron (float): Minimum LEBRON impact score.
        min_salary (int): Minimum salary in millions.
        max_salary (int): Maximum salary in millions.
        search_query (str): Player name search query.
        season_data (dict): Season data store containing selected season.

    Returns:
        tuple: A tuple containing 4 Plotly figures and 3 HTML table components.
    """
    # Get data for the selected season
    season = season_data.get('season', CURRENT_SEASON) if season_data else CURRENT_SEASON
    df_season, _ = get_current_data(season)
    
    if df_season.empty:
        # Return empty figures/tables if no data
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark', paper_bgcolor='#0f1623')
        return empty_fig, empty_fig, empty_fig, empty_fig, html.Div(), html.Div(), html.Div()
    
    # Filter the DataFrame based on slider inputs
    # We convert millions back to raw dollars for comparison
    filtered_df = df_season[
        (df_season['LEBRON'] >= min_lebron) & 
        (df_season['current_year_salary'] >= min_salary * 1_000_000) & 
        (df_season['current_year_salary'] <= max_salary * 1_000_000)
    ].copy()
    
    # Recalculate value metrics for the filtered subset
    # This ensures percentiles are relative to the currently selected pool of players
    # Don't save to DB for filtered views
    if 'current_year_salary' in filtered_df.columns and 'LEBRON' in filtered_df.columns:
        valid_salary = filtered_df['current_year_salary'].dropna()
        valid_lebron = filtered_df['LEBRON'].dropna()
        
        if len(valid_salary) > 0 and len(valid_lebron) > 0:
            salary_min = valid_salary.min()
            salary_max = valid_salary.max()
            filtered_df['salary_norm'] = 100 * (filtered_df['current_year_salary'] - salary_min) / (salary_max - salary_min)
            
            lebron_min = valid_lebron.min()
            lebron_max = valid_lebron.max()
            filtered_df['impact_norm'] = 100 * (filtered_df['LEBRON'] - lebron_min) / (lebron_max - lebron_min)
            
            filtered_df['value_gap'] = filtered_df['impact_norm']*1.4 - filtered_df['salary_norm']*0.9 - 10
    
    # Generate updated visualizations
    fig_scatter = visualizations.create_salary_impact_scatter(filtered_df)
    fig_under = visualizations.create_underpaid_bar(filtered_df)
    fig_beeswarm = visualizations.create_player_beeswarm(filtered_df)
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
    
    return fig_scatter, fig_under, fig_beeswarm, fig_over, table_under, table_over, table_all


# ============================================================================
# DIAMOND FINDER CALLBACKS
# ============================================================================

# Store for the similarity model (built once per session)
_diamond_finder_model = {}

def _get_diamond_finder_model(season):
    """Get or build the Diamond Finder similarity model for a season."""
    if season not in _diamond_finder_model:
        # Load and prepare data
        result = data_processing.load_and_merge_data(season=season, from_db=True)
        df = result[0] if isinstance(result, tuple) else result
        df = data_processing.calculate_player_value_metrics(df, season=season)
        
        # Build archetype-based similarity model
        # Pass season for fetching advanced stats from NBA API
        model, scaler, df_filtered, feature_info = data_processing.build_current_season_similarity(df, season=season)
        
        if model is not None:
            _diamond_finder_model[season] = {
                'model': model,
                'scaler': scaler,
                'df': df_filtered,
                'feature_info': feature_info
            }
    
    return _diamond_finder_model.get(season)


@app.callback(
    Output('diamond-finder-player', 'options'),
    [Input('season-data-store', 'data')]
)
def populate_diamond_finder_dropdown(season_data):
    """Populate the Diamond Finder dropdown with player options."""
    season = season_data.get('season', CURRENT_SEASON) if season_data else CURRENT_SEASON
    
    model_data = _get_diamond_finder_model(season)
    if model_data is None:
        return []
    
    df = model_data['df']
    
    # Create options sorted by salary (highest first) - these are the players you'd want to replace
    df_sorted = df.sort_values('current_year_salary', ascending=False)
    
    options = []
    for _, row in df_sorted.iterrows():
        name = row['player_name']
        salary = row.get('current_year_salary', 0)
        lebron = row.get('LEBRON', 0)
        options.append({
            'label': f"{name} - ${salary/1e6:.1f}M (LEBRON: {lebron:.2f})",
            'value': name
        })
    
    return options


@app.callback(
    [Output('diamond-finder-results', 'children'),
     Output('diamond-finder-target-info', 'children')],
    [Input('diamond-finder-player', 'value'),
     Input('season-data-store', 'data')]
)
def update_diamond_finder(player_name, season_data):
    """Find and display replacement players for the selected player."""
    if not player_name:
        return html.Div([
            html.P("Select a player above to find cheaper replacements.", 
                   className="text-muted text-center py-4"),
            html.P("Higher-paid players will have more replacement options.", 
                   className="text-muted text-center", style={"fontSize": "12px"})
        ]), html.Div()
    
    season = season_data.get('season', CURRENT_SEASON) if season_data else CURRENT_SEASON
    
    model_data = _get_diamond_finder_model(season)
    if model_data is None:
        return html.Div("Error loading similarity model", className="text-danger"), html.Div()
    
    model = model_data['model']
    scaler = model_data['scaler']
    df = model_data['df']
    feature_info = model_data['feature_info']
    
    # Get target player info
    target_mask = df['player_name'] == player_name
    if not target_mask.any():
        return html.Div(f"Player '{player_name}' not found", className="text-warning"), html.Div()
    
    target_row = df[target_mask].iloc[0]
    target_salary = target_row.get('current_year_salary', 0)
    target_lebron = target_row.get('LEBRON', 0)
    target_archetype = target_row.get('Offensive Archetype', 'Unknown')
    target_defense = target_row.get('Defensive Role', 'Unknown')
    target_id = target_row.get('PLAYER_ID')
    
    # Find replacements
    replacements = data_processing.find_replacement_players(
        player_name, df, model, scaler, feature_info, max_results=8
    )
    
    # Create results display
    results = visualizations.create_diamond_finder_results(
        replacements, player_name, target_salary, target_lebron
    )
    
    # Create target info card
    img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(target_id)}.png" if pd.notna(target_id) else ""
    
    target_info = html.Div([
        dbc.Row([
            dbc.Col([
                html.Img(src=img_url, style={
                    "width": "50px", "height": "50px", "borderRadius": "50%",
                    "border": "2px solid #ff6b35", "objectFit": "cover"
                }) if img_url else None
            ], width="auto"),
            dbc.Col([
                html.Div(player_name, style={"fontWeight": "700", "color": "#ff6b35"}),
                html.Div(f"${target_salary/1e6:.1f}M | LEBRON: {target_lebron:.2f}", 
                        style={"fontSize": "12px", "color": "#adb5bd"}),
                html.Div([
                    html.Span(target_archetype, style={
                        "backgroundColor": "rgba(45, 150, 199, 0.3)",
                        "color": "#2D96C7",
                        "padding": "1px 6px",
                        "borderRadius": "4px",
                        "fontSize": "10px",
                        "marginRight": "4px"
                    }),
                    html.Span(target_defense, style={
                        "color": "#6c757d",
                        "fontSize": "10px"
                    })
                ])
            ])
        ], align="center")
    ], style={
        "backgroundColor": "rgba(255, 107, 53, 0.1)",
        "padding": "10px",
        "borderRadius": "8px",
        "border": "1px solid rgba(255, 107, 53, 0.3)"
    })
    
    return results, target_info




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


# ============================================================================
# LINEUP CHEMISTRY CALLBACKS
# ============================================================================

@app.callback(
    [Output('chart-best-lineups', 'figure'),
     Output('chart-worst-lineups', 'figure'),
     Output('chart-lineup-scatter', 'figure'),
     Output('table-best-lineups', 'children'),
     Output('table-worst-lineups', 'children')],
    [Input('lineup-team-dropdown', 'value'),
     Input('lineup-size-radio', 'value'),
     Input('lineup-min-minutes', 'value')]
)
def update_lineup_analysis(team_abbr, group_size, min_minutes):
    """
    Updates all lineup visualizations based on user filters.
    
    This callback is triggered when the user changes the team, lineup size,
    or minimum minutes filter. It fetches new data from the NBA API (or cache)
    and regenerates all charts and tables.
    
    Args:
        team_abbr (str): Team abbreviation or 'ALL' for league-wide.
        group_size (int): 2 for duos, 3 for trios.
        min_minutes (int): Minimum minutes played together.
        
    Returns:
        tuple: Best chart, worst chart, scatter chart, best table, worst table.
    """
    try:
        # Handle 'ALL' teams selection
        team_filter = None if team_abbr == 'ALL' else team_abbr
        
        print(f"Fetching lineup data: team={team_filter}, size={group_size}, min_min={min_minutes}")
        
        # Get best and worst lineups
        df_best = data_processing.get_best_lineups(
            team_abbr=team_filter,
            group_quantity=group_size,
            min_minutes=min_minutes,
            top_n=15
        )
        
        df_worst = data_processing.get_worst_lineups(
            team_abbr=team_filter,
            group_quantity=group_size,
            min_minutes=min_minutes,
            top_n=15
        )
        
        # Create visualizations
        fig_best = visualizations.create_lineup_bar_chart(
            df_best, 
            title='Best Lineups', 
            color='#06d6a0',
            metric='PLUS_MINUS'
        )
        
        fig_worst = visualizations.create_lineup_bar_chart(
            df_worst, 
            title='Worst Lineups', 
            color='#ef476f',
            metric='PLUS_MINUS'
        )
        
        fig_scatter = visualizations.create_lineup_scatter(df_best, df_worst)
        
        # Create tables
        table_best = visualizations.create_lineup_table(df_best, table_type='best')
        table_worst = visualizations.create_lineup_table(df_worst, table_type='worst')
        
        return fig_best, fig_worst, fig_scatter, table_best, table_worst
        
    except Exception as e:
        print(f"Error updating lineup analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty figures on error
        empty_fig = go.Figure().add_annotation(text=f"Error loading data: {str(e)}")
        empty_fig.update_layout(template='plotly_dark', paper_bgcolor='#0f1623', height=400)
        
        return empty_fig, empty_fig, empty_fig, html.P(f"Error: {str(e)}"), html.P(f"Error: {str(e)}")





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
