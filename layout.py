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
from data_processing import TEAM_ABBR_MAP

# Create reverse mapping for display
ABBR_TO_NAME = {v: k for k, v in TEAM_ABBR_MAP.items()}
# Add missing teams if any
ABBR_TO_NAME.update({
    'PHX': 'Phoenix Suns', 'BKN': 'Brooklyn Nets', 'CHA': 'Charlotte Hornets', 
    'NOP': 'New Orleans Pelicans', 'UTA': 'Utah Jazz'
})

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
                            style_cell={'backgroundColor': '#0f1623', 'color': '#e4e6eb', 'textAlign': 'left', 'padding': '12px', 'fontSize': '13px', 'border': '1px solid #2c3e50'},
                            style_header={'backgroundColor': '#151b26', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #ff6b35', 'color': '#e4e6eb'},
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
                
                # View Selection Dropdown
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Analysis View:", className="text-center d-block mb-2", 
                                  style={"color": "#e4e6eb", "fontWeight": "600"}),
                        dbc.RadioItems(
                            id='view-selector',
                            options=[
                                {'label': 'Player Analysis', 'value': 'player'},
                                {'label': 'Team Analysis', 'value': 'team'}
                            ],
                            value='player',
                            inline=True,
                            className="btn-group",
                            inputClassName="btn-check",
                            labelClassName="btn btn-outline-primary",
                            labelCheckedClassName="active",
                            style={"display": "flex", "justifyContent": "center", "gap": "10px"}
                        )
                    ], width={"size": 4, "offset": 4})
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
