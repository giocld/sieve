"""
Layout module for Sieve NBA Analytics.
This module defines the HTML structure and CSS styling of the dashboard.
It organizes the application into tabs, rows, and columns using Dash Bootstrap Components.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np

from .data_processing import TEAM_ABBR_MAP
from .config import get_season_display, CURRENT_SEASON
from .cache_manager import cache

# Create reverse mapping for display
ABBR_TO_NAME = {v: k for k, v in TEAM_ABBR_MAP.items()}


def _get_season_options():
    """Get available seasons for the dropdown."""
    # Get seasons that have LEBRON data
    try:
        seasons = cache.list_lebron_seasons()
        if not seasons:
            seasons = [CURRENT_SEASON]
    except:
        seasons = [CURRENT_SEASON]
    
    # Sort descending (most recent first)
    seasons = sorted(seasons, reverse=True)
    
    return [{'label': s, 'value': s} for s in seasons]


# Global Style Constants
SECTION_TITLE_STYLE = {
    "color": "#ffffff",
    "fontWeight": "700",
    "fontSize": "24px",
    "textAlign": "left",
    "letterSpacing": "0.5px",
    "marginBottom": "8px"
}

CARD_HEADER_TEXT_STYLE = {
    "fontWeight": "600",
    "fontSize": "14px",
    "color": "#e4e6eb",
    "letterSpacing": "0.3px"
}

CARD_HEADER_BG_STYLE = {
    "backgroundColor": "#0a0e14",
    "borderBottom": "1px solid #1e2a3a",
    "padding": "12px 16px"
}

def create_landing_tab(df=None, df_teams=None):
    """
    Creates the landing page with quick stats, navigation cards and metric explanations.
    
    Args:
        df: Player dataframe for stats
        df_teams: Team dataframe for stats
    
    Returns:
        dbc.Container: The landing page layout.
    """
    # Calculate quick stats
    num_players = len(df) if df is not None and not df.empty else 0
    num_teams = len(df_teams) if df_teams is not None and not df_teams.empty else 0
    avg_salary = df['current_year_salary'].mean() / 1_000_000 if df is not None and 'current_year_salary' in df.columns else 0
    avg_lebron = df['LEBRON'].mean() if df is not None and 'LEBRON' in df.columns else 0
    total_payroll = df_teams['Total_Payroll'].sum() / 1_000_000_000 if df_teams is not None and 'Total_Payroll' in df_teams.columns else 0
    
    # Find top performers
    top_value_player = ""
    top_value_gap = 0
    if df is not None and 'value_gap' in df.columns and not df.empty:
        top_idx = df['value_gap'].idxmax()
        top_value_player = df.loc[top_idx, 'player_name'] if 'player_name' in df.columns else "N/A"
        top_value_gap = df.loc[top_idx, 'value_gap']
    
    most_efficient_team = ""
    if df_teams is not None and 'Efficiency_Index' in df_teams.columns and not df_teams.empty:
        top_team_idx = df_teams['Efficiency_Index'].idxmax()
        most_efficient_team = df_teams.loc[top_team_idx, 'Abbrev'] if 'Abbrev' in df_teams.columns else "N/A"
    
    return dbc.Container([
        # Header Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Overview", style={
                        "color": "#ffffff",
                        "fontWeight": "700",
                        "fontSize": "32px",
                        "marginBottom": "8px",
                        "letterSpacing": "0.5px"
                    }),
                    html.P("NBA Player Value & Efficiency Analysis Platform", style={
                        "color": "#6c757d",
                        "fontSize": "14px",
                        "marginBottom": "0"
                    })
                ], className="pt-4 pb-2")
            ])
        ]),
        
        # Quick Stats Row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span(f"{num_players}", style={"color": "#ff6b35", "fontWeight": "700", "fontSize": "28px"}),
                    html.Div("Players Tracked", style={"color": "#4a5568", "fontSize": "11px", "fontWeight": "500", "marginTop": "-4px"})
                ], style={"textAlign": "center"})
            ], xs=6, md=2, className="mb-3"),
            dbc.Col([
                html.Div([
                    html.Span(f"{num_teams}", style={"color": "#06d6a0", "fontWeight": "700", "fontSize": "28px"}),
                    html.Div("Teams", style={"color": "#4a5568", "fontSize": "11px", "fontWeight": "500", "marginTop": "-4px"})
                ], style={"textAlign": "center"})
            ], xs=6, md=2, className="mb-3"),
            dbc.Col([
                html.Div([
                    html.Span(f"${avg_salary:.1f}M", style={"color": "#2D96C7", "fontWeight": "700", "fontSize": "28px"}),
                    html.Div("Avg Salary", style={"color": "#4a5568", "fontSize": "11px", "fontWeight": "500", "marginTop": "-4px"})
                ], style={"textAlign": "center"})
            ], xs=6, md=2, className="mb-3"),
            dbc.Col([
                html.Div([
                    html.Span(f"{avg_lebron:+.2f}", style={"color": "#ffd166", "fontWeight": "700", "fontSize": "28px"}),
                    html.Div("Avg LEBRON", style={"color": "#4a5568", "fontSize": "11px", "fontWeight": "500", "marginTop": "-4px"})
                ], style={"textAlign": "center"})
            ], xs=6, md=2, className="mb-3"),
            dbc.Col([
                html.Div([
                    html.Span(f"${total_payroll:.1f}B", style={"color": "#ef476f", "fontWeight": "700", "fontSize": "28px"}),
                    html.Div("League Payroll", style={"color": "#4a5568", "fontSize": "11px", "fontWeight": "500", "marginTop": "-4px"})
                ], style={"textAlign": "center"})
            ], xs=6, md=2, className="mb-3"),
            dbc.Col([
                html.Div([
                    html.Span(most_efficient_team if most_efficient_team else "N/A", style={"color": "#06d6a0", "fontWeight": "700", "fontSize": "28px"}),
                    html.Div("Most Efficient", style={"color": "#4a5568", "fontSize": "11px", "fontWeight": "500", "marginTop": "-4px"})
                ], style={"textAlign": "center"})
            ], xs=6, md=2, className="mb-3"),
        ], className="py-3 mb-2", style={"backgroundColor": "#0a0e14", "borderRadius": "8px", "border": "1px solid #1e2a3a"}),
        
        # Top Performer Highlight
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("Best Value: ", style={"color": "#4a5568", "fontSize": "12px"}),
                    html.Span(top_value_player, style={"color": "#06d6a0", "fontSize": "12px", "fontWeight": "600"}),
                    html.Span(f" (+{top_value_gap:.1f} gap)", style={"color": "#6c757d", "fontSize": "11px"}) if top_value_gap > 0 else None
                ], style={"textAlign": "center", "marginBottom": "16px"})
            ])
        ]) if top_value_player else None,
        
        # Quick Navigation Cards
        html.H6("Quick Access", style={"color": "#4a5568", "fontWeight": "600", "fontSize": "11px", 
                                        "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Span("Players", style={"color": "#ff6b35", "fontWeight": "700", "fontSize": "16px"})
                            ], className="mb-2"),
                            html.P("Find underpaid gems and overpaid contracts. Compare salary vs. on-court impact.", 
                                   style={"color": "#6c757d", "fontSize": "12px", "marginBottom": "12px", "lineHeight": "1.5"}),
                            html.Div([
                                html.Span("Value Gap", style={"color": "#06d6a0", "fontSize": "11px", "fontWeight": "600"}),
                                html.Span(" | ", style={"color": "#2d3748"}),
                                html.Span("LEBRON", style={"color": "#ff6b35", "fontSize": "11px", "fontWeight": "600"}),
                                html.Span(" | ", style={"color": "#2d3748"}),
                                html.Span("WAR", style={"color": "#2D96C7", "fontSize": "11px", "fontWeight": "600"}),
                            ])
                        ], style={"padding": "16px"})
                    ], style={"backgroundColor": "#12171f", "border": "1px solid #1e2a3a", "cursor": "pointer"}, className="h-100")
                ], id="card-nav-player", n_clicks=0, className="h-100")
            ], md=3, className="mb-3"),
            
            dbc.Col([
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Span("Teams", style={"color": "#06d6a0", "fontWeight": "700", "fontSize": "16px"})
                            ], className="mb-2"),
                            html.P("Evaluate team efficiency. Which franchises are maximizing their payroll investment?", 
                                   style={"color": "#6c757d", "fontSize": "12px", "marginBottom": "12px", "lineHeight": "1.5"}),
                            html.Div([
                                html.Span("Efficiency Index", style={"color": "#ffd166", "fontSize": "11px", "fontWeight": "600"}),
                                html.Span(" | ", style={"color": "#2d3748"}),
                                html.Span("Cost/Win", style={"color": "#ef476f", "fontSize": "11px", "fontWeight": "600"}),
                            ])
                        ], style={"padding": "16px"})
                    ], style={"backgroundColor": "#12171f", "border": "1px solid #1e2a3a", "cursor": "pointer"}, className="h-100")
                ], id="card-nav-team", n_clicks=0, className="h-100")
            ], md=3, className="mb-3"),
            
            dbc.Col([
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Span("Lineups", style={"color": "#2D96C7", "fontWeight": "700", "fontSize": "16px"})
                            ], className="mb-2"),
                            html.P("Discover which duos and trios dominate together or struggle when sharing minutes.", 
                                   style={"color": "#6c757d", "fontSize": "12px", "marginBottom": "12px", "lineHeight": "1.5"}),
                            html.Div([
                                html.Span("Plus/Minus", style={"color": "#2D96C7", "fontSize": "11px", "fontWeight": "600"}),
                                html.Span(" | ", style={"color": "#2d3748"}),
                                html.Span("Minutes Together", style={"color": "#6c757d", "fontSize": "11px", "fontWeight": "600"}),
                            ])
                        ], style={"padding": "16px"})
                    ], style={"backgroundColor": "#12171f", "border": "1px solid #1e2a3a", "cursor": "pointer"}, className="h-100")
                ], id="card-nav-lineup", n_clicks=0, className="h-100")
            ], md=3, className="mb-3"),
            
            dbc.Col([
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Span("Comps", style={"color": "#ffd166", "fontWeight": "700", "fontSize": "16px"})
                            ], className="mb-2"),
                            html.P("Find historical statistical matches. Who does a player compare to from past seasons?", 
                                   style={"color": "#6c757d", "fontSize": "12px", "marginBottom": "12px", "lineHeight": "1.5"}),
                            html.Div([
                                html.Span("20+ Features", style={"color": "#ff6b35", "fontSize": "11px", "fontWeight": "600"}),
                                html.Span(" | ", style={"color": "#2d3748"}),
                                html.Span("10 Years Data", style={"color": "#6c757d", "fontSize": "11px", "fontWeight": "600"}),
                            ])
                        ], style={"padding": "16px"})
                    ], style={"backgroundColor": "#12171f", "border": "1px solid #1e2a3a", "cursor": "pointer"}, className="h-100")
                ], id="card-nav-similarity", n_clicks=0, className="h-100")
            ], md=3, className="mb-3"),
        ], className="mb-4"),
        
        # Divider
        html.Hr(style={"borderColor": "#1e2a3a", "margin": "16px 0 24px 0"}),
        
        # Metrics Reference Section
        html.H6("Metrics Reference", style={"color": "#4a5568", "fontWeight": "600", "fontSize": "11px", 
                                             "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}),
        
        # Primary Metrics Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("LEBRON", style={"color": "#ff6b35", "fontWeight": "700", "fontSize": "15px"}),
                        ], className="mb-1"),
                        html.P("Luck-adjusted Estimate of Box score Rating On-court / Off-court Net", 
                               style={"color": "#4a5568", "fontSize": "10px", "fontStyle": "italic", "marginBottom": "8px"}),
                        html.P("Points per 100 possessions added vs. replacement level.", 
                               style={"color": "#adb5bd", "fontSize": "11px", "marginBottom": "10px"}),
                        html.Div([
                            html.Span("+3.0+ All-Star", style={"color": "#06d6a0", "fontSize": "10px", "marginRight": "8px"}),
                            html.Span("+1 to +3 Starter", style={"color": "#2D96C7", "fontSize": "10px", "marginRight": "8px"}),
                            html.Span("<-1 Below Avg", style={"color": "#ef476f", "fontSize": "10px"}),
                        ])
                    ], style={"padding": "14px"})
                ], style={"backgroundColor": "#12171f", "border": "1px solid #1e2a3a", "borderTop": "2px solid #ff6b35"})
            ], lg=3, md=6, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("Value Gap", style={"color": "#06d6a0", "fontWeight": "700", "fontSize": "15px"}),
                        ], className="mb-1"),
                        html.P("Normalized Impact - Normalized Salary", 
                               style={"color": "#4a5568", "fontSize": "10px", "fontStyle": "italic", "marginBottom": "8px"}),
                        html.P("Difference between contribution and cost, scaled 0-100.", 
                               style={"color": "#adb5bd", "fontSize": "11px", "marginBottom": "10px"}),
                        html.Div([
                            html.Span("+ Underpaid", style={"color": "#06d6a0", "fontSize": "10px", "marginRight": "8px"}),
                            html.Span("0 Fair", style={"color": "#ffd166", "fontSize": "10px", "marginRight": "8px"}),
                            html.Span("- Overpaid", style={"color": "#ef476f", "fontSize": "10px"}),
                        ])
                    ], style={"padding": "14px"})
                ], style={"backgroundColor": "#12171f", "border": "1px solid #1e2a3a", "borderTop": "2px solid #06d6a0"})
            ], lg=3, md=6, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("Plus/Minus", style={"color": "#2D96C7", "fontWeight": "700", "fontSize": "15px"}),
                        ], className="mb-1"),
                        html.P("Point differential while on floor together", 
                               style={"color": "#4a5568", "fontSize": "10px", "fontStyle": "italic", "marginBottom": "8px"}),
                        html.P("Total points scored minus allowed for lineup combos.", 
                               style={"color": "#adb5bd", "fontSize": "11px", "marginBottom": "10px"}),
                        html.Div([
                            html.Span("+100 Elite", style={"color": "#06d6a0", "fontSize": "10px", "marginRight": "8px"}),
                            html.Span("+ Good", style={"color": "#2D96C7", "fontSize": "10px", "marginRight": "8px"}),
                            html.Span("- Poor", style={"color": "#ef476f", "fontSize": "10px"}),
                        ])
                    ], style={"padding": "14px"})
                ], style={"backgroundColor": "#12171f", "border": "1px solid #1e2a3a", "borderTop": "2px solid #2D96C7"})
            ], lg=3, md=6, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("Efficiency Index", style={"color": "#ffd166", "fontWeight": "700", "fontSize": "15px"}),
                        ], className="mb-1"),
                        html.P("(2x Wins Z-Score) - Payroll Z-Score", 
                               style={"color": "#4a5568", "fontSize": "10px", "fontStyle": "italic", "marginBottom": "8px"}),
                        html.P("Team wins generated relative to spending.", 
                               style={"color": "#adb5bd", "fontSize": "11px", "marginBottom": "10px"}),
                        html.Div([
                            html.Span("+ Efficient", style={"color": "#06d6a0", "fontSize": "10px", "marginRight": "8px"}),
                            html.Span("- Inefficient", style={"color": "#ef476f", "fontSize": "10px"}),
                        ])
                    ], style={"padding": "14px"})
                ], style={"backgroundColor": "#12171f", "border": "1px solid #1e2a3a", "borderTop": "2px solid #ffd166"})
            ], lg=3, md=6, className="mb-3"),
        ]),
        
        # Data Info
        html.Div([
            html.P([
                html.Span("Data: ", style={"color": "#4a5568", "fontWeight": "600"}),
                html.Span(get_season_display(), style={"color": "#6c757d"}),
                html.Span(" | ", style={"color": "#2d3748"}),
                html.Span("Source: ", style={"color": "#4a5568", "fontWeight": "600"}),
                html.Span("BBall Index LEBRON, NBA API, Basketball Reference", style={"color": "#6c757d"}),
            ], style={"fontSize": "11px", "textAlign": "center", "marginTop": "16px", "marginBottom": "0"})
        ])
        
    ], fluid=True, style={"padding": "0 24px"})


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
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Player Value Analysis", style={
                        "color": "#ffffff",
                        "fontWeight": "700",
                        "fontSize": "28px",
                        "marginBottom": "4px",
                        "letterSpacing": "0.5px"
                    }),
                    html.P("Identify underpaid and overpaid players based on impact vs. salary", style={
                        "color": "#6c757d",
                        "fontSize": "13px",
                        "marginBottom": "0"
                    })
                ], className="pt-3 pb-3")
            ])
        ]),
        
        # Diamond Finder Row: KNN-based replacement finder (FIRST)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Diamond Finder", className="mb-0 d-inline", style=CARD_HEADER_TEXT_STYLE),
                        html.Span(" - Find Cheaper Replacements", style={"color": "#06d6a0", "fontSize": "12px", "marginLeft": "8px"})
                    ], style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([
                        dbc.Alert([
                            html.Strong("How it works: "),
                            "Select a player to find statistically similar alternatives at a lower cost. ",
                            "Uses KNN similarity (8 features) + archetype/position filtering."
                        ], color="success", className="mb-2 py-2", style={"fontSize": "12px", "backgroundColor": "rgba(6, 214, 160, 0.15)", "border": "1px solid #06d6a0", "color": "#e4e6eb"}),
                        
                        # Player selector dropdown
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select player to replace:", style={"color": "#adb5bd", "fontSize": "13px", "marginBottom": "4px"}),
                                dcc.Dropdown(
                                    id='diamond-finder-player',
                                    placeholder="Choose a player...",
                                    style={
                                        'fontSize': '13px',
                                    },
                                    className="dark-dropdown"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Div(id='diamond-finder-target-info', className="mt-4")
                            ], md=6)
                        ], className="mb-3"),
                        
                        # Results container
                        html.Div(id='diamond-finder-results', style={"minHeight": "200px"})
                    ], style={"padding": "15px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], xs=12, className="mb-4")
        ]),
        
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
        
        # Charts Row: Scatter Plot and Beeswarm - EQUAL WIDTH
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Salary vs Impact", className="mb-0 d-inline", style=CARD_HEADER_TEXT_STYLE),
                    ], style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([
                        dbc.Alert([
                            html.Strong("How to read: "),
                            "X-axis = LEBRON impact score | Y-axis = Salary | ",
                            "Size = WAR (total wins added) | Color = Value Gap (green=underpaid, red=overpaid). ",
                            "Top-left = high impact, low salary (best value)."
                        ], color="info", className="mb-2 py-2", style={"fontSize": "12px", "backgroundColor": "rgba(45, 150, 199, 0.15)", "border": "1px solid #2D96C7", "color": "#e4e6eb"}),
                        dcc.Graph(id='chart-salary-impact', style={"height": "500px", "width": "100%"}, config={"displayModeBar": False, "responsive": True})
                    ], style={"padding": "10px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50", "height": "100%"})
            ], xs=12, lg=6, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("All Players by Impact", className="mb-0 d-inline", style=CARD_HEADER_TEXT_STYLE),
                    ], style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([
                        dbc.Alert([
                            html.Strong("Player Impact Map: "),
                            "Every player shown by LEBRON impact. ",
                            "Green border = underpaid, Red border = overpaid. Hover for details."
                        ], color="info", className="mb-2 py-2", style={"fontSize": "12px", "backgroundColor": "rgba(45, 150, 199, 0.15)", "border": "1px solid #2D96C7", "color": "#e4e6eb"}),
                        dcc.Graph(id='chart-beeswarm', style={"height": "500px", "width": "100%"}, config={"displayModeBar": False, "responsive": True})
                    ], style={"padding": "10px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50", "height": "100%"})
            ], xs=12, lg=6, className="mb-4")
        ], className="g-3"),
        
        # Value Analysis Row: Underpaid and Overpaid side by side
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Best Value (Underpaid)", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([
                        dcc.Graph(id='chart-underpaid'),
                        html.Hr(style={"borderColor": "#2c3e50", "margin": "16px 0"}),
                        html.Div(id='table-underpaid', style={"maxHeight": "300px", "overflowY": "auto"})
                    ], style={"padding": "12px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], xs=12, lg=6, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Worst Value (Overpaid)", className="mb-0", 
                                          style=CARD_HEADER_TEXT_STYLE),
                                  style=CARD_HEADER_BG_STYLE),
                    dbc.CardBody([
                        dcc.Graph(id='chart-overpaid'),
                        html.Hr(style={"borderColor": "#2c3e50", "margin": "16px 0"}),
                        html.Div(id='table-overpaid', style={"maxHeight": "300px", "overflowY": "auto"})
                    ], style={"padding": "12px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"})
            ], xs=12, lg=6, className="mb-4")
        ]),
        
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
    ], fluid=True, className="content-container")


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
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Team Analysis", style={
                        "color": "#ffffff",
                        "fontWeight": "700",
                        "fontSize": "28px",
                        "marginBottom": "4px",
                        "letterSpacing": "0.5px"
                    }),
                    html.P("Compare team efficiency, payroll spending, and performance metrics", style={
                        "color": "#6c757d",
                        "fontSize": "13px",
                        "marginBottom": "0"
                    })
                ], className="pt-3 pb-3")
            ])
        ]),
        
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
                            dbc.Col([
                                html.Label("Team 1", className="text-center d-block", style={"fontSize": "11px", "color": "#ff6b35", "fontWeight": "600", "marginBottom": "3px"}),
                                dbc.Select(
                                    id='team-radar-dropdown-1',
                                    options=[{'label': ABBR_TO_NAME.get(t, t), 'value': t} for t in sorted(df_teams['Abbrev'].unique())] if not df_teams.empty else [],
                                    value='BOS' if not df_teams.empty and 'BOS' in df_teams['Abbrev'].values else (df_teams['Abbrev'].iloc[0] if not df_teams.empty else None),
                                    class_name="bg-dark text-white border-secondary",
                                    style={'fontSize': '13px', 'borderColor': '#ff6b35'}
                                )
                            ], xs=12, md=5, className="mb-2 mb-md-0"),
                            dbc.Col([
                                html.Label("Team 2", className="text-center d-block", style={"fontSize": "11px", "color": "#2D96C7", "fontWeight": "600", "marginBottom": "3px"}),
                                dbc.Select(
                                    id='team-radar-dropdown-2',
                                    options=[{'label': ABBR_TO_NAME.get(t, t), 'value': t} for t in sorted(df_teams['Abbrev'].unique())] if not df_teams.empty else [],
                                    value='LAL' if not df_teams.empty and 'LAL' in df_teams['Abbrev'].values else (df_teams['Abbrev'].iloc[1] if len(df_teams) > 1 else None),
                                    class_name="bg-dark text-white border-secondary",
                                    style={'fontSize': '13px', 'borderColor': '#2D96C7'}
                                )
                            ], xs=12, md=5)
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
    ], fluid=True, className="content-container")


def create_main_layout():
    """
    Creates the main application shell with professional tabbed navigation.

    Returns:
        html.Div: The root HTML element of the application.
    """
    
    # Professional navigation bar with dropdown menu
    navbar = dbc.Navbar(
        dbc.Container([
            # Brand - clickable to go home
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("SIEVE", style={
                            "fontWeight": "800", 
                            "fontSize": "24px", 
                            "color": "#ffffff",
                            "letterSpacing": "3px"
                        }),
                        html.Span(" Analytics", style={
                            "fontWeight": "300", 
                            "fontSize": "14px", 
                            "color": "#6c757d",
                            "marginLeft": "8px",
                            "textTransform": "uppercase",
                            "letterSpacing": "2px"
                        })
                    ], id="nav-brand", style={"cursor": "pointer"})
                ], width="auto")
            ], align="center", className="g-0"),
            
            # Horizontal Navigation + Season Selector
            dbc.Row([
                dbc.Col([
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Overview", id="nav-home", href="#", active=True, className="nav-pill-custom")),
                        dbc.NavItem(dbc.NavLink("Players", id="nav-player", href="#", className="nav-pill-custom")),
                        dbc.NavItem(dbc.NavLink("Teams", id="nav-team", href="#", className="nav-pill-custom")),
                        dbc.NavItem(dbc.NavLink("Lineups", id="nav-lineup", href="#", className="nav-pill-custom")),
                        dbc.NavItem(dbc.NavLink("Comps", id="nav-similarity", href="#", className="nav-pill-custom")),
                    ], pills=True, className="nav-pills-dark")
                ], width="auto"),
                # Season Selector
                dbc.Col([
                    dcc.Dropdown(
                        id='season-selector',
                        options=_get_season_options(),
                        value=CURRENT_SEASON,
                        clearable=False,
                        style={
                            "width": "110px",
                            "fontSize": "13px",
                        },
                        className="dark-dropdown"
                    )
                ], width="auto", className="ms-3")
            ], align="center", className="g-0 ms-auto"),
        ], fluid=True),
        color="#0a0e14",
        dark=True,
        sticky="top",
        style={"borderBottom": "1px solid #1e2a3a", "padding": "12px 0"}
    )
    
    # Hidden store for view selection (replaces dcc.Tabs)
    view_store = dcc.Store(id='view-selector', data='home')
    
    # Store for quick navigation from cards (will be updated by card clicks)
    nav_request_store = dcc.Store(id='nav-request', data=None)
    
    # Store for season-specific data (will be populated by callback)
    season_data_store = dcc.Store(id='season-data-store', data={'season': CURRENT_SEASON})
    
    return html.Div([
        view_store,
        nav_request_store,
        season_data_store,
        navbar,
        
        # Content Area
        html.Div(id='page-content', style={
            "minHeight": "calc(100vh - 120px)",
            "padding": "0"
        }),
        
        # Footer
        html.Div([
            dbc.Container([
                html.Hr(style={"borderColor": "#1e2a3a", "opacity": "0.5", "margin": "0"}),
                dbc.Row([
                    dbc.Col([
                        html.P("Sieve Analytics", style={
                            "color": "#4a5568", 
                            "fontSize": "12px", 
                            "fontWeight": "600",
                            "marginBottom": "4px"
                        }),
                        html.P("NBA Player Value & Efficiency Analysis", style={
                            "color": "#2d3748", 
                            "fontSize": "11px",
                            "marginBottom": "0"
                        })
                    ], md=6, className="text-start"),
                    dbc.Col([
                        html.P([
                            html.A("GitHub", href="https://github.com/giocld/sieve", target="_blank", 
                                   style={"color": "#4a5568", "textDecoration": "none", "fontSize": "11px"}),
                            html.Span(" | ", style={"color": "#2d3748"}),
                            html.Span("Built with Dash", style={"color": "#2d3748", "fontSize": "11px"})
                        ], className="mb-0", style={"textAlign": "right"})
                    ], md=6, className="text-end d-flex align-items-center justify-content-end")
                ], className="py-3", align="center")
            ], fluid=True)
        ], style={"backgroundColor": "#0a0e14"})
        
    ], style={"backgroundColor": "#0f1623", "minHeight": "100vh"})

def create_lineup_tab(team_options):
    """
    Creates the layout for the 'Lineup Chemistry' tab.
    
    This tab allows users to analyze which duos and trios perform best/worst
    when playing together on the floor.
    
    Args:
        team_options (list): List of team dropdown options.
        
    Returns:
        dbc.Container: The complete layout container for the lineup tab.
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Lineup Chemistry", style={
                        "color": "#ffffff",
                        "fontWeight": "700",
                        "fontSize": "28px",
                        "marginBottom": "4px",
                        "letterSpacing": "0.5px"
                    }),
                    html.P("Analyze duo and trio performance when sharing the floor", style={
                        "color": "#6c757d",
                        "fontSize": "13px",
                        "marginBottom": "0"
                    })
                ], className="pt-3 pb-3")
            ])
        ]),
        
        # Controls Card
        dbc.Card([
            dbc.CardHeader(html.H5("Analysis Controls", className="mb-0", 
                                  style=CARD_HEADER_TEXT_STYLE),
                          style=CARD_HEADER_BG_STYLE),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Team:", className="fw-bold mb-2", 
                                  style={"fontSize": "14px", "color": "#e4e6eb"}),
                        dbc.Select(
                            id='lineup-team-dropdown',
                            options=[{'label': 'League-Wide (All Teams)', 'value': 'ALL'}] + team_options,
                            value='ALL',
                            class_name="bg-dark text-white border-secondary",
                            style={'fontSize': '13px'}
                        )
                    ], xs=12, md=4, className="mb-3 mb-md-0"),
                    
                    dbc.Col([
                        html.Label("Lineup Size:", className="fw-bold mb-2", 
                                  style={"fontSize": "14px", "color": "#e4e6eb"}),
                        dbc.RadioItems(
                            id='lineup-size-radio',
                            options=[
                                {'label': '  Duos (2-man)', 'value': 2},
                                {'label': '  Trios (3-man)', 'value': 3}
                            ],
                            value=2,
                            inline=True,
                            className="text-white",
                            inputStyle={"marginRight": "5px"},
                            labelStyle={"marginRight": "20px"}
                        )
                    ], xs=12, md=4, className="mb-3 mb-md-0"),
                    
                    dbc.Col([
                        html.Label("Min. Minutes Together:", className="fw-bold mb-2", 
                                  style={"fontSize": "14px", "color": "#e4e6eb"}),
                        dcc.Slider(
                            id='lineup-min-minutes',
                            min=10,
                            max=500,
                            value=100,
                            step=10,
                            marks={10: '10', 50: '50', 100: '100', 200: '200', 300: '300', 500: '500'},
                            tooltip={"placement": "bottom", "always_visible": False}
                        )
                    ], xs=12, md=4)
                ])
            ])
        ], className="mb-4", style={"backgroundColor": "#1a2332", "border": "1px solid #2c3e50"}),
        
        # Info Alert
        dbc.Alert([
            html.Strong("How to read: "),
            "Plus/Minus (+/-) = total point differential when these players share the floor. ",
            "Positive = outscored opponents | Negative = got outscored. ",
            "MIN = total minutes played together over the season."
        ], color="info", className="mb-4 py-2", style={"fontSize": "12px", "backgroundColor": "rgba(45, 150, 199, 0.15)", "border": "1px solid #2D96C7", "color": "#e4e6eb"}),
        
        # Tabbed view for Best/Worst
        dbc.Tabs([
            dbc.Tab(label="Top Performers", tab_id="best-tab", children=[
                dbc.Card([
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id='chart-best-lineups'),
                            type="circle",
                            color="#06d6a0"
                        ),
                        html.Hr(style={"borderColor": "#2c3e50"}),
                        html.H6("Detailed Statistics", className="mb-3", style={"color": "#06d6a0", "fontWeight": "600"}),
                        html.Div(id='table-best-lineups', style={"maxHeight": "400px", "overflowY": "auto"})
                    ], style={"padding": "20px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #1e2a3a", "borderTop": "3px solid #06d6a0"})
            ], label_style={"color": "#06d6a0", "fontWeight": "500"}, 
               active_label_style={"backgroundColor": "#06d6a0", "color": "#0f1623", "fontWeight": "600"}),
            
            dbc.Tab(label="Underperformers", tab_id="worst-tab", children=[
                dbc.Card([
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id='chart-worst-lineups'),
                            type="circle",
                            color="#ef476f"
                        ),
                        html.Hr(style={"borderColor": "#2c3e50"}),
                        html.H6("Detailed Statistics", className="mb-3", style={"color": "#ef476f", "fontWeight": "600"}),
                        html.Div(id='table-worst-lineups', style={"maxHeight": "400px", "overflowY": "auto"})
                    ], style={"padding": "20px"})
                ], style={"backgroundColor": "#1a2332", "border": "1px solid #1e2a3a", "borderTop": "3px solid #ef476f"})
            ], label_style={"color": "#ef476f", "fontWeight": "500"}, 
               active_label_style={"backgroundColor": "#ef476f", "color": "#0f1623", "fontWeight": "600"}),
        ], id="lineup-tabs", active_tab="best-tab", className="mb-4"),
        
        # Hidden graph for callback (not displayed)
        html.Div(dcc.Graph(id='chart-lineup-scatter', style={'display': 'none'}), style={'display': 'none'})
        
    ], fluid=True, className="content-container")


def create_similarity_tab(player_options):
    """
    Creates the layout for the 'Historical Comps' tab.
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Historical Comparisons", style={
                        "color": "#ffffff",
                        "fontWeight": "700",
                        "fontSize": "28px",
                        "marginBottom": "4px",
                        "letterSpacing": "0.5px"
                    }),
                    html.P("Find statistical matches from the past decade based on production and playstyle", style={
                        "color": "#6c757d",
                        "fontSize": "13px",
                        "marginBottom": "0"
                    })
                ], className="pt-3 pb-3")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Search Parameters", className="mb-3", style={"color": "#ffffff", "fontWeight": "600"}),
                        html.Label("Player", className="mb-2", style={"color": "#6c757d", "fontSize": "12px", "fontWeight": "500"}),
                        dcc.Dropdown(
                            id='similarity-player-dropdown',
                            options=player_options,
                            placeholder="Search players...",
                            className="mb-3",
                            style={'color': '#000'}
                        ),
                        html.Label("Season", className="mb-2", style={"color": "#6c757d", "fontSize": "12px", "fontWeight": "500"}),
                        dcc.Dropdown(
                            id='similarity-season-dropdown',
                            placeholder="Select season...",
                            className="mb-3",
                            style={'color': '#000'}
                        ),
                        dcc.Checklist(
                            id='similarity-exclude-self',
                            options=[{'label': ' Exclude player from results', 'value': 'exclude'}],
                            value=['exclude'],
                            className="mb-3",
                            style={"color": "#adb5bd", "fontSize": "13px"},
                            inputStyle={"marginRight": "8px"}
                        ),
                        html.P("Returns the top 5 statistical matches based on 20+ weighted features including production, efficiency, and playstyle metrics.",
                               style={"color": "#4a5568", "fontSize": "12px", "marginBottom": "0"})
                    ], style={"padding": "24px"})
                ], style={"backgroundColor": "#12171f", "border": "1px solid #1e2a3a"})
            ], md=6, className="offset-md-3 mb-4")
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='similarity-results-container')
            ], md=12)
        ])
    ], fluid=True, className="content-container")

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
                        style={"margin": "0", "padding": "0", "height": "220px", "width": "100%"}
                    )
                ], style={"height": "220px", "overflow": "hidden"})
            ], style={"padding": "15px", "display": "flex", "flexDirection": "column"})
        ], style={
            "backgroundColor": bg_color, 
            "border": f"2px solid {border_color}",
            "height": "550px",  # FIXED height instead of min-height
            "boxShadow": "0 2px 4px rgba(0,0,0,0.3)"
        }),
        width=12, lg=4, className="mb-4"
    )
