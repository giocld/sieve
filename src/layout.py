"""
Layout module for Sieve NBA Analytics.
This module defines the HTML structure and CSS styling of the dashboard.
It organizes the application into tabs, rows, and columns using Dash Bootstrap Components.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .data_processing import TEAM_ABBR_MAP

# Create reverse mapping for display
ABBR_TO_NAME = {v: k for k, v in TEAM_ABBR_MAP.items()}


# Global Style Constants
SECTION_TITLE_STYLE = {
    "color": "#e9ecef",
    "fontWeight": "600",
    "fontSize": "28px",
    "textAlign": "center"
}

CARD_HEADER_TEXT_STYLE = {
    "fontWeight": "600",
    "fontSize": "16px",
    "color": "#e4e6eb"
}

CARD_HEADER_BG_STYLE = {
    "backgroundColor": "#151b26",
    "borderBottom": "2px solid #ff6b35"
}

def create_player_tab(df):
    """
    Creates the layout for the 'Player Analysis' tab.
    
    This tab contains:
    1. Filter Controls (Sliders for Salary and Impact).
    2. A grid of 4 charts (Salary vs Impact, Underpaid, Age Curve, Overpaid).
    3. Detailed data tables for Underpaid/Overpaid players.
    4. A comprehensive table of all players.

    Args:
        df (pd.DataFrame): The player DataFrame, used to determine slider ranges.

    Returns:
        dbc.Container: The complete layout container for the player tab.
    """
    
    # Calculate dynamic slider ranges based on the data
    max_salary_m = int(df['current_year_salary'].max() / 1000000) if 'current_year_salary' in df.columns else 50
    min_lebron = float(df['LEBRON'].min())
    max_lebron = float(df['LEBRON'].max())
    
    return dbc.Container([
        html.H2("Player Value Analysis", className="mt-4 mb-4", 
                style=SECTION_TITLE_STYLE),
        
        # Filter Controls Card
        # Contains sliders to filter the dataset by Salary and LEBRON Impact
        dbc.Card([
            dbc.CardHeader(html.H5("Filters", className="mb-0", style=CARD_HEADER_TEXT_STYLE),
                          style=CARD_HEADER_BG_STYLE),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Min Salary ($M):", className="fw-bold mb-2", style={"fontSize": "14px"}),
                        dcc.Slider(
                            id='min-salary',
                            min=0,
                            max=max_salary_m,
                            value=5,
                            marks={i: f"${i}" for i in range(0, max_salary_m + 1, 5)},
                            tooltip={"placement": "bottom", "always_visible": False}
                        )
                    ], xs=12, md=6),
                    
                    dbc.Col([
                        html.Label("Max Salary ($M):", className="fw-bold mb-2", style={"fontSize": "14px"}),
                        dcc.Slider(
                            id='max-salary',
                            min=0,
                            max=max_salary_m,
                            value=max_salary_m,
                            marks={i: f"${i}" for i in range(0, max_salary_m + 1, 5)},
                            tooltip={"placement": "bottom", "always_visible": False}
                        )
                    ], xs=12, md=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Min LEBRON Impact:", className="fw-bold mb-2", style={"fontSize": "14px"}),
                        dcc.Slider(
                            id='min-lebron',
                            min=min_lebron,
                            max=max_lebron,
                            value=min_lebron,
                            step=0.5,
                            marks={round(i, 1): f"{i:.1f}" for i in np.arange(np.floor(min_lebron * 2) / 2, np.ceil(max_lebron * 2) / 2 + 0.1, 0.5)},
                            tooltip={"placement": "bottom", "always_visible": False}
                        )
                    ], md=12)
                ])
            ])
        ], className="mb-4", style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"}),
        
        # Charts Row 1: Scatter Plot and Underpaid Bar Chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Salary vs Impact", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([dcc.Graph(id='chart-salary-impact')], style={"padding": "10px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], xs=12, lg=6, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Age vs Impact Curve", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([dcc.Graph(id='chart-off-def')], style={"padding": "10px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], xs=12, lg=6, className="mb-4")
        ]),
        
        # Charts Row 2: Age Curve and Overpaid Bar Chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Top Underpaid Players", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([dcc.Graph(id='chart-underpaid')], style={"padding": "10px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], xs=12, lg=6, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Top Overpaid Players", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([dcc.Graph(id='chart-overpaid')], style={"padding": "10px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], xs=12, lg=6, className="mb-4")
        ]),
        
        # Tables Row: Side-by-side lists of Top 10 Underpaid and Overpaid
        dbc.Row([
            dbc.Col([
                html.H5("Top Underpaid Players (Best Value)", className="mb-3", 
                       style={"fontWeight": "600", "color": "#06d6a0"}),
                html.Div(id='table-underpaid', style={"maxHeight": "300px", "overflowY": "auto"})
            ], md=6),
            dbc.Col([
                html.H5("Top Overpaid Players (Worst Value)", className="mb-3", 
                       style={"fontWeight": "600", "color": "#ef476f"}),
                html.Div(id='table-overpaid', style={"maxHeight": "300px", "overflowY": "auto"})
            ], md=6)
        ], className="mb-4"),
        
        # All Players Table Row: Full dataset listing
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                html.H5(f"All {len(df)} Players (Sorted by Value Gap)", className="mb-0", 
                                       style=CARD_HEADER_TEXT_STYLE)
                            ], width=6),
                            dbc.Col([
                                dbc.Input(
                                    id='player-search-input',
                                    type='text',
                                    placeholder='Search players...',
                                    className='mb-0',
                                    style={
                                        'backgroundColor': '#0f1623',
                                        'color': '#e4e6eb',
                                        'border': '1px solid #2c3e50',
                                        'borderRadius': '4px',
                                        'fontSize': '13px',
                                        'padding': '6px 12px'
                                    }
                                )
                            ], width=6)
                        ], align='center')
                    ], style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([
                        html.Div(id='table-all-players', style={"maxHeight": "800px", "overflowY": "auto"})
                    ], style={"padding": "0px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ])
        ], className="mb-4")
    ], fluid=True, style={"backgroundColor": "#0f1623", "padding": "30px 20px"})


def create_team_tab(df_teams, fig_quadrant, fig_grid):
    """
    Creates the layout for the 'Team Analysis' tab.
    
    This tab contains:
    1. Efficiency Quadrant Chart (Scatter Plot).
    2. Team Efficiency Grid (Heatmap).
    3. Detailed Team Statistics Table.

    Args:
        df_teams (pd.DataFrame): Team metrics DataFrame.
        fig_quadrant (go.Figure): Pre-generated Quadrant chart.
        fig_grid (go.Figure): Pre-generated Grid chart.

    Returns:
        dbc.Container: The complete layout container for the team tab.
    """
    
    return dbc.Container([
        html.H2("Team Analysis", className="mt-4 mb-4", 
                style=SECTION_TITLE_STYLE),
        
        # Charts Row: Quadrant and Treemap side-by-side
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Efficiency Quadrant", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([dcc.Graph(figure=fig_quadrant)], style={"padding": "10px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], lg=6, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Team Payroll vs Efficiency", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([dcc.Graph(figure=fig_grid)], style={"padding": "10px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], lg=6, className="mb-4")
        ]),
        
        # Radar Chart Section - Team Comparison
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("Team Comparison Radar", className="mb-0 text-center", 
                                  style=CARD_HEADER_TEXT_STYLE),
                            html.P("Compare team strengths and weaknesses", 
                                  className="mb-3 text-center", 
                                  style={"fontSize": "12px", "color": "#6c757d"})
                        ]),
                        dbc.Row([
                            dbc.Col(width=2),
                            dbc.Col([
                                html.Label("Team 1", className="text-center d-block", style={"fontSize": "11px", "color": "#ff6b35", "fontWeight": "600", "marginBottom": "3px"}),
                                dbc.Select(
                                    id='team-radar-dropdown-1',
                                    options=[{'label': ABBR_TO_NAME.get(t, t), 'value': t} for t in sorted(df_teams['Abbrev'].unique())] if not df_teams.empty else [],
                                    value='BOS' if not df_teams.empty and 'BOS' in df_teams['Abbrev'].values else (df_teams['Abbrev'].iloc[0] if not df_teams.empty else None),
                                    class_name="bg-dark text-white border-secondary",
                                    style={'fontSize': '13px', 'borderColor': '#ff6b35'}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Team 2", className="text-center d-block", style={"fontSize": "11px", "color": "#2D96C7", "fontWeight": "600", "marginBottom": "3px"}),
                                dbc.Select(
                                    id='team-radar-dropdown-2',
                                    options=[{'label': ABBR_TO_NAME.get(t, t), 'value': t} for t in sorted(df_teams['Abbrev'].unique())] if not df_teams.empty else [],
                                    value='LAL' if not df_teams.empty and 'LAL' in df_teams['Abbrev'].values else (df_teams['Abbrev'].iloc[1] if len(df_teams) > 1 else None),
                                    class_name="bg-dark text-white border-secondary",
                                    style={'fontSize': '13px', 'borderColor': '#2D96C7'}
                                )
                            ], width=4),
                            dbc.Col(width=2)
                        ], justify="center")
                    ], style={"backgroundColor": "#151b26", "borderBottom": "2px solid #ff6b35", "padding": "15px"}),
                    dbc.CardBody([
                        dcc.Graph(id='chart-team-radar')
                    ], style={"padding": "15px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], lg=12, className="mb-4")
        ]),
        
        # Table Row: Comprehensive Team Stats
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Team Statistics", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([
                        dash_table.DataTable(
                            data=df_teams.assign(
                                Logo_Markdown=lambda x: x['Logo_URL'].apply(lambda url: f"![Logo]({url})" if pd.notna(url) else "")
                            ).sort_values('Efficiency_Index', ascending=False).to_dict('records') if not df_teams.empty else [],
                            columns=[
                                {'name': 'Logo', 'id': 'Logo_Markdown', 'presentation': 'markdown'},
                                {'name': 'Team', 'id': 'Abbrev'},
                                {'name': 'Wins', 'id': 'WINS'},
                                {'name': 'Losses', 'id': 'LOSSES'},
                                {'name': 'Payroll', 'id': 'Payroll_Display'},
                                {'name': 'Cost/Win', 'id': 'CPW_Display'},
                                {'name': 'Eff Index', 'id': 'Efficiency_Index', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Total WAR', 'id': 'Total_WAR', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            style_table={'overflowX': 'auto'},
                            style_cell={'backgroundColor': '#0f1623', 'color': '#e4e6eb', 'textAlign': 'left', 'padding': '4px', 'fontSize': '11px', 'border': '1px solid #2c3e50'},
                            style_header={'backgroundColor': '#151b26', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #ff6b35', 'color': '#e4e6eb', 'padding': '4px'},
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#1a2332'}
                            ],
                            sort_action='native',
                            page_action='none',  # Show all teams without pagination
                            markdown_options={'html': True}
                        )
                    ], style={"padding": "15px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], width=12)
        ])
    ], fluid=True, style={"backgroundColor": "#0f1623", "padding": "30px 20px"})


def create_main_layout():
    """
    Creates the main application shell.
    
    This function wraps the individual tabs in a top-level container, adding
    the application header and the view selection dropdown.

    Returns:
        html.Div: The root HTML element of the application.
    """
    return html.Div([
        # Header Section (Hero)
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("Sieve", className="display-3 text-center mb-2", 
                               style={"fontWeight": "700", "color": "#e9ecef", "letterSpacing": "2px"}),
                        html.P("NBA Player Value Analytics", 
                               className="text-center mb-4", 
                               style={"color": "#adb5bd", "fontSize": "16px", "fontWeight": "300"})
                    ])
                ]),
                
                # Navigation Tabs
                dbc.Row([
                    dbc.Col([
                        dcc.Tabs(
                            id='view-selector',
                            value='player',
                            children=[
                                dcc.Tab(label='Player Analysis', value='player', className='custom-tab', selected_className='custom-tab--selected'),
                                dcc.Tab(label='Team Analysis', value='team', className='custom-tab', selected_className='custom-tab--selected'),
                                dcc.Tab(label='Historical Comps', value='similarity', className='custom-tab', selected_className='custom-tab--selected'),
                            ],
                            colors={
                                "border": "#2c3e50",
                                "primary": "#ff6b35",
                                "background": "#0f1623"
                            },
                            parent_className="custom-tabs",
                            className="custom-tabs-container"
                        )
                    ], width=12)
                ])
            ], fluid=True)
        ], className="hero-header"),
        
        # Content Area
        html.Div(id='page-content', style={"minHeight": "calc(100vh - 350px)"}),
        
        # Footer
        html.Div([
            dbc.Container([
                html.Hr(style={"borderColor": "#2c3e50", "opacity": "0.5"}),
                html.P([
                    "Sieve Analytics © 2025 | ",
                    html.A("Documentation", href="https://github.com/giocld/sieve", target="_blank", style={"color": "#2D96C7", "textDecoration": "none"}),
                    " | ",
                    html.Span("Built with Dash & Plotly", style={"color": "#adb5bd"})
                ], className="text-center mt-4 mb-4", style={"color": "#6c757d", "fontSize": "14px"})
            ], fluid=True)
        ])
        
    ], style={"backgroundColor": "#0f1623", "minHeight": "100vh", "display": "flex", "flexDirection": "column"})

def create_similarity_tab(player_options):
    """
    Creates the layout for the 'Historical Comps' tab.
    """
    return dbc.Container([
        html.H2("Historical Comps", className="mt-4 mb-4", 
                style=SECTION_TITLE_STYLE),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Find Historical Comps", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([
                        html.Label("Select a Player:", className="fw-bold mb-2", style={"color": "#e4e6eb"}),
                        dcc.Dropdown(
                            id='similarity-player-dropdown',
                            options=player_options,
                            placeholder="Type to search...",
                            className="mb-3",
                            style={'color': '#000'}
                        ),
                        html.Label("Select a Season:", className="fw-bold mb-2", style={"color": "#e4e6eb"}),
                        dcc.Dropdown(
                            id='similarity-season-dropdown',
                            placeholder="First select a player...",
                            className="mb-3",
                            style={'color': '#000'}
                        ),
                        dcc.Checklist(
                            id='similarity-exclude-self',
                            options=[{'label': ' Exclude Player from Results', 'value': 'exclude'}],
                            value=['exclude'],
                            className="mb-3 text-white",
                            inputStyle={"marginRight": "5px"}
                        ),
                        html.P("Finds the top 5 statistical matches from the last 10 years based on production and playstyle.",
                               className="text-muted small")
                    ], style={"padding": "20px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], md=6, className="offset-md-3 mb-4")
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='similarity-results-container')
            ], md=12)
        ])
    ], fluid=True, style={"backgroundColor": "#0f1623", "padding": "30px 20px"})

def create_similarity_card(name, season, pid, stats, position='Wing', match_score=None, distance=None, is_target=False):
    """
    Creates a styled card for a player in the similarity engine with hexagonal radar chart.
    
    Args:
        name (str): Player name.
        season (str): Season ID (e.g., '2024-25').
        pid (int): Player ID for fetching the headshot.
        stats (dict): Dictionary of player statistics.
        position (str): Player position group ('Guard', 'Wing', 'Big').
        match_score (float, optional): Match score (0-100, non-linear). Defaults to None.
        distance (float, optional): Raw cosine distance for debugging. Defaults to None.
        is_target (bool, optional): Whether this is the target player (gold styling). Defaults to False.

    Returns:
        dbc.Col: A column containing the player card.
    """
    from . import visualizations
    
    # Styling based on type
    if is_target:
        border_color = "#ff6b35"
        bg_color = "rgba(255, 107, 53, 0.1)"
        badge_text = "SELECTED PLAYER"
        badge_color = "warning"
        match_display = None
        radar_color = "#ff6b35" # Force Orange for target
    else:
        # Color-code match score - APPLY TO ENTIRE CARD
        if match_score and match_score >= 90:
            score_color = "#06d6a0"  # Green - Excellent match
            badge_color = "success"
            match_quality = "EXCELLENT"
        elif match_score and match_score >= 75:
            score_color = "#2D96C7"  # Blue - Good match
            badge_color = "info"
            match_quality = "GOOD"
        elif match_score and match_score >= 60:
            score_color = "#ffd60a"  # Yellow - Moderate match
            badge_color = "warning"
            match_quality = "MODERATE"
        else:
            score_color = "#ef476f"  # Red - Weak match
            badge_color = "danger"
            match_quality = "WEAK"
        
        border_color = score_color
        bg_color = "#151b26"
        radar_color = score_color # Force Radar to match card color
        badge_text = f"{match_quality} MATCH"
        
        # Build match display with score and optional distance
        match_display = html.Div([
            # Match Score (large, prominent)
            html.Div([
                html.Span(f"{match_score:.0f}", style={
                    "fontSize": "2.5rem", 
                    "fontWeight": "800",
                    "color": score_color,
                    "lineHeight": "1"
                }),
                html.Span("/100", style={
                    "fontSize": "1.2rem",
                    "color": "#6c757d",
                    "marginLeft": "4px"
                })
            ], className="text-center mb-2"),
            
            # Visual progress bar
            html.Div([
                html.Div(style={
                    "width": f"{match_score}%",
                    "height": "6px",
                    "backgroundColor": score_color,
                    "borderRadius": "3px",
                    "transition": "width 0.3s ease"
                })
            ], style={
                "width": "100%",
                "height": "6px",
                "backgroundColor": "rgba(255,255,255,0.1)",
                "borderRadius": "3px",
                "marginBottom": "8px"
            }),
            
            # Distance (small, technical detail)
            html.Small(f"cosine distance: {distance:.4f}" if distance is not None else "",
                       className="text-muted", 
                       style={"fontSize": "0.65rem", "fontStyle": "italic"})
        ], className="mb-3")

    img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
    
    # Create mini radar chart
    radar_fig = visualizations.create_player_radar_mini(stats, position=radar_color, player_name=name)


    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                # Top Badge
                html.Div([
                    html.Span(badge_text, className="badge mb-2", style={
                        "fontSize": "0.8rem", 
                        "letterSpacing": "1px",
                        "backgroundColor": radar_color,
                        "color": "#fff"
                    })
                ], className="text-center", style={"minHeight": "30px"}),
                
                # Match Score Display (for comparison players only) - fixed height
                html.Div([
                    match_display if match_display else html.Div()
                ], style={"minHeight": "110px"}),  # Always 110px for alignment
                
                # Image & Info - fixed height
                html.Div([
                    html.Img(src=img_url, style={
                        "width": "90px", "height": "90px", 
                        "borderRadius": "50%", "objectFit": "cover",
                        "border": f"3px solid {border_color}", "marginBottom": "10px"
                    }),
                    html.H5(name, className="text-white mb-0", style={"fontWeight": "700", "fontSize": "0.95rem"}),
                    html.Small(season, style={"fontSize": "0.75rem", "color": radar_color, "fontWeight": "600"}),
                    html.Br(),
                    # Position badge
                    html.Span(position, className="badge mt-1", style={
                        "fontSize": "0.65rem",
                        "backgroundColor": "rgba(255,255,255,0.1)",
                        "color": "#adb5bd",
                        "fontWeight": "500"
                    })
                ], className="text-center mb-2", style={"minHeight": "150px"}),
                
                # Hexagonal Radar Chart - FIXED HEIGHT
                html.Div([
                    dcc.Graph(
                        figure=radar_fig,
                        config={'displayModeBar': False},
                        style={"margin": "0", "padding": "0", "height": "180px", "width": "100%"}
                    )
                ], style={"height": "180px", "overflow": "hidden"})
            ], style={"padding": "15px", "display": "flex", "flexDirection": "column"})
        ], style={
            "backgroundColor": bg_color, 
            "border": f"2px solid {border_color}",
            "height": "550px",  # FIXED height instead of min-height
            "boxShadow": "0 2px 4px rgba(0,0,0,0.3)"
        }),
        width=12, lg=4, className="mb-4"
    )
