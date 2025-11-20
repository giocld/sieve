import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import os


# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

print("Loading and preparing data...")

lebron_file = 'LEBRON.csv'
contracts_file = 'basketball_reference_contracts.csv'
standings_file = 'nba_standings_cache.csv'

TEAM_ABBR_MAP = {
    'Cleveland Cavaliers': 'CLE', 'Oklahoma City Thunder': 'OKC', 'Boston Celtics': 'BOS',
    'Houston Rockets': 'HOU', 'Los Angeles Lakers': 'LAL', 'New York Knicks': 'NYK',
    'Indiana Pacers': 'IND', 'Denver Nuggets': 'DEN', 'LA Clippers': 'LAC',
    'Milwaukee Bucks': 'MIL', 'Detroit Pistons': 'DET', 'Minnesota Timberwolves': 'MIN',
    'Orlando Magic': 'ORL', 'Golden State Warriors': 'GSW', 'Memphis Grizzlies': 'MEM',
    'Atlanta Hawks': 'ATL', 'Chicago Bulls': 'CHI', 'Sacramento Kings': 'SAC',
    'Dallas Mavericks': 'DAL', 'Miami Heat': 'MIA', 'Phoenix Suns': 'PHX',
    'Toronto Raptors': 'TOR', 'Portland Trail Blazers': 'POR', 'Brooklyn Nets': 'BKN',
    'Philadelphia 76ers': 'PHI', 'San Antonio Spurs': 'SAS', 'Charlotte Hornets': 'CHA',
    'New Orleans Pelicans': 'NOP', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

# ==========================================
# 2. PLAYER DATA PROCESSING
# ==========================================

if not os.path.exists(lebron_file) or not os.path.exists(contracts_file):
    print(f"ERROR: Missing {lebron_file} or {contracts_file}")
    exit(1)

df_lebron = pd.read_csv(lebron_file)
df_contracts = pd.read_csv(contracts_file)

df_lebron = df_lebron.rename(columns={'Player': 'player_name'})

for col in ['LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON']:
    if col in df_lebron.columns:
        df_lebron[col] = pd.to_numeric(df_lebron[col], errors='coerce')

if 'Offensive Archetype' in df_lebron.columns and 'Defensive Role' in df_lebron.columns:
    df_lebron['archetype'] = (
        df_lebron['Offensive Archetype'].fillna('Unknown').astype(str) + 
        ' / ' + 
        df_lebron['Defensive Role'].fillna('Unknown').astype(str)
    )
else:
    df_lebron['archetype'] = 'Unknown'

df = pd.merge(df_lebron, df_contracts, on='player_name', how='inner')
df = df.drop_duplicates(subset=['player_name'], keep='first')

if 'current_year_salary' in df.columns:
    nan_mask = df['current_year_salary'].isna()
    if 'average_annual_value' in df.columns:
        df.loc[nan_mask, 'current_year_salary'] = df.loc[nan_mask, 'average_annual_value']

if 'current_year_salary' in df.columns and 'LEBRON' in df.columns:
    valid_salary = df['current_year_salary'].dropna()
    valid_lebron = df['LEBRON'].dropna()
    
    if len(valid_salary) > 0 and len(valid_lebron) > 0:
        salary_min = valid_salary.min()
        salary_max = valid_salary.max()
        df['salary_norm'] = 100 * (df['current_year_salary'] - salary_min) / (salary_max - salary_min)
        
        lebron_min = valid_lebron.min()
        lebron_max = valid_lebron.max()
        df['impact_norm'] = 100 * (df['LEBRON'] - lebron_min) / (lebron_max - lebron_min)
        
        df['value_gap'] = df['impact_norm'] - df['salary_norm']
    else:
        df['value_gap'] = 0
else:
    df['value_gap'] = 0

print(f"Loaded {len(df)} players")


# ==========================================
# 3. TEAM DATA PROCESSING
# ==========================================

print("Processing Team Efficiency metrics...")

df_standings = pd.read_csv(standings_file)
df_standings.columns = df_standings.columns.str.strip()

if 'FullName' not in df_standings.columns and 'TeamCity' in df_standings.columns and 'TeamName' in df_standings.columns:
    df_standings['FullName'] = df_standings['TeamCity'] + ' ' + df_standings['TeamName']

df_standings['Abbrev'] = df_standings['FullName'].map(TEAM_ABBR_MAP)
df_standings = df_standings.dropna(subset=['Abbrev'])

abbr_map = {'PHO': 'PHX', 'CHO': 'CHA', 'BRK': 'BKN', 'NOH': 'NOP', 'TOT': 'UNK'}
if 'Team(s)' in df_lebron.columns:
    df['Team_Abbr'] = df['Team(s)'].replace(abbr_map)
    df['Team_Main'] = df['Team_Abbr'].apply(lambda x: str(x).split('/')[0].strip() if isinstance(x, str) else 'UNK')
else:
    df['Team_Main'] = 'UNK'

df_teams = df.groupby('Team_Main').agg({
    'current_year_salary': 'sum',
    'LEBRON WAR': 'sum',
    'LEBRON': 'mean',
    'player_name': 'count'
}).reset_index()

df_teams = df_teams.rename(columns={
    'current_year_salary': 'Total_Payroll',
    'LEBRON WAR': 'Total_WAR',
    'Team_Main': 'Abbrev'
})

df_teams = pd.merge(df_teams, df_standings[['Abbrev', 'WINS', 'LOSSES']], on='Abbrev', how='left')
df_teams = df_teams[(df_teams['Total_Payroll'] > 0) & (df_teams['Abbrev'] != 'UNK') & (df_teams['WINS'].notna())].copy()

if not df_teams.empty:
    df_teams['Cost_Per_Win'] = df_teams.apply(
        lambda row: row['Total_Payroll'] / row['WINS'] if row['WINS'] > 0 else 0, axis=1
    )
    
    wins_std = df_teams['WINS'].std()
    pay_std = df_teams['Total_Payroll'].std()
    
    if wins_std > 0 and pay_std > 0:
        z_wins = (df_teams['WINS'] - df_teams['WINS'].mean()) / wins_std
        z_pay = (df_teams['Total_Payroll'] - df_teams['Total_Payroll'].mean()) / pay_std
        df_teams['Efficiency_Index'] = z_wins - z_pay
    else:
        df_teams['Efficiency_Index'] = 0

df_teams['Payroll_Display'] = df_teams['Total_Payroll'].apply(lambda x: f"${x/1_000_000:.1f}M")
df_teams['CPW_Display'] = df_teams['Cost_Per_Win'].apply(lambda x: f"${x/1_000_000:.2f}M" if x > 0 else "N/A")

print(f"Loaded {len(df_teams)} teams")


# ==========================================
# 4. BUILD TEAM VISUALIZATIONS
# ==========================================

if not df_teams.empty:
    avg_payroll = df_teams['Total_Payroll'].mean()
    avg_wins = df_teams['WINS'].mean()
    df_teams['WAR_Size'] = df_teams['Total_WAR'].clip(lower=0.5)

    fig_quadrant = px.scatter(
        df_teams, x='Total_Payroll', y='WINS',
        text='Abbrev', color='Efficiency_Index', 
        size='WAR_Size',
        hover_data=['Total_WAR', 'LOSSES'],
        color_continuous_scale='RdYlGn',
        title='<b>Efficiency Quadrant: Wins vs. Payroll</b>',
        labels={'Total_Payroll': 'Total Payroll ($)', 'WINS': 'Wins', 'Efficiency_Index': 'Eff Index'},
        template='plotly_dark'
    )
    fig_quadrant.update_layout(height=480, margin=dict(l=80, r=80, t=70, b=70))
    fig_quadrant.add_vline(x=avg_payroll, line_dash="dash", line_color="gray")
    fig_quadrant.add_hline(y=avg_wins, line_dash="dash", line_color="gray")
    fig_quadrant.update_traces(textposition='top center')

    df_roi_sorted = df_teams.sort_values('Efficiency_Index', ascending=True)
    fig_roi = px.bar(
        df_roi_sorted, x='Efficiency_Index', y='Abbrev',
        orientation='h', color='Efficiency_Index',
        color_continuous_scale='RdYlGn',
        title='<b>Team Efficiency Index</b>',
        template='plotly_dark'
    )
    fig_roi.update_layout(height=480, margin=dict(l=80, r=80, t=70, b=70), showlegend=False)
else:
    fig_quadrant = go.Figure().add_annotation(text="No Team Data Available")
    fig_quadrant.update_layout(height=480)
    fig_roi = go.Figure().add_annotation(text="No Team Data Available")
    fig_roi.update_layout(height=480)


# ==========================================
# 5. CREATE APP & LAYOUT
# ==========================================

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server


# --- PLAYER TAB LAYOUT ---
player_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Player Analysis", className="mt-3 mb-4", style={"color": "#e9ecef", "fontWeight": "700", "fontSize": "24px"})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Min Salary ($M):", className="fw-bold mb-2"),
            dcc.Slider(
                id='min-salary',
                min=0,
                max=int(df['current_year_salary'].max() / 1000000) if 'current_year_salary' in df.columns else 50,
                value=0,
                marks={i: f"${i}" for i in range(0, int(df['current_year_salary'].max() / 1000000) + 1 if 'current_year_salary' in df.columns else 51, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], md=6, className="mb-4"),
        
        dbc.Col([
            html.Label("Max Salary ($M):", className="fw-bold mb-2"),
            dcc.Slider(
                id='max-salary',
                min=0,
                max=int(df['current_year_salary'].max() / 1000000) if 'current_year_salary' in df.columns else 50,
                value=int(df['current_year_salary'].max() / 1000000) if 'current_year_salary' in df.columns else 50,
                marks={i: f"${i}" for i in range(0, int(df['current_year_salary'].max() / 1000000) + 1 if 'current_year_salary' in df.columns else 51, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], md=6, className="mb-4")
    ], className="mb-3"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Min LEBRON Impact:", className="fw-bold mb-2"),
            dcc.Slider(
                id='min-lebron',
                min=float(df['LEBRON'].min()),
                max=float(df['LEBRON'].max()),
                value=float(df['LEBRON'].min()),
                marks={round(i, 1): f"{i:.1f}" for i in np.arange(
                    float(df['LEBRON'].min()),
                    float(df['LEBRON'].max()) + 0.5, 0.5
                )},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], md=12, className="mb-5")
    ]),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='chart-salary-impact')], md=6),
        dbc.Col([dcc.Graph(id='chart-underpaid')], md=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='chart-off-def')], md=6),
        dbc.Col([dcc.Graph(id='chart-overpaid')], md=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H5("Top Underpaid (Green = Best Value)", className="text-success mt-3 mb-2"),
            html.Div(id='table-underpaid', style={"maxHeight": "300px", "overflowY": "auto"})
        ], md=6),
        dbc.Col([
            html.H5("Top Overpaid (Red = Worst Value)", className="text-danger mt-3 mb-2"),
            html.Div(id='table-overpaid', style={"maxHeight": "300px", "overflowY": "auto"})
        ], md=6)
    ], className="mb-5"),
    
    dbc.Row([
        dbc.Col([
            html.H5(f"All {len(df)} Players (Sorted by Value Gap)", className="mt-3 mb-2", style={"color": "#e9ecef"}),
            html.Div(id='table-all-players', style={"maxHeight": "800px", "overflowY": "auto"})
        ])
    ], className="mb-5")
], fluid=True, style={"backgroundColor": "#1e1e1e", "padding": "20px"})


# --- TEAM TAB LAYOUT ---
team_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Team Efficiency Analysis", className="mt-3 mb-4", style={"color": "#e9ecef", "fontWeight": "700", "fontSize": "24px"})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_quadrant)
        ], lg=6, className="mb-4"),
        
        dbc.Col([
            dcc.Graph(figure=fig_roi)
        ], lg=6, className="mb-4")
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H5("Team Statistics", className="mb-3", style={"color": "#e9ecef"}),
            dash_table.DataTable(
                data=df_teams.sort_values('Efficiency_Index', ascending=False).to_dict('records') if not df_teams.empty else [],
                columns=[
                    {'name': 'Team', 'id': 'Abbrev'},
                    {'name': 'Wins', 'id': 'WINS'},
                    {'name': 'Losses', 'id': 'LOSSES'},
                    {'name': 'Payroll', 'id': 'Payroll_Display'},
                    {'name': 'Cost/Win', 'id': 'CPW_Display'},
                    {'name': 'Eff Index', 'id': 'Efficiency_Index', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Total WAR', 'id': 'Total_WAR', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                ],
                style_table={'overflowX': 'auto', 'maxHeight': '500px', 'overflowY': 'auto'},
                style_cell={'backgroundColor': '#222', 'color': 'white', 'textAlign': 'left', 'padding': '10px', 'fontSize': '13px'},
                style_header={'backgroundColor': '#444', 'fontWeight': 'bold', 'textAlign': 'center'},
                sort_action='native',
                filter_action='native',
                page_size=15
            )
        ], width=12)
    ])
], fluid=True, style={"backgroundColor": "#1e1e1e", "padding": "20px"})


# --- MAIN APP LAYOUT WITH TABS ---
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Sieve", className="display-4 text-center mt-4 mb-1", style={"fontWeight": "700", "color": "#e9ecef"}),
                html.H4("NBA Archetypes: Salary, Impact & Value", className="text-center mb-5", style={"color": "#adb5bd", "fontWeight": "400", "fontSize": "14px"})
            ])
        ]),
    ], fluid=True),
    
    dcc.Tabs(
        id='main-tabs',
        value='tab-players',
        children=[
            dcc.Tab(
                label='Player Analysis',
                value='tab-players',
                children=player_tab,
                style={
                    'padding': '0px',
                    'fontWeight': 'bold',
                    'backgroundColor': '#1e1e1e',
                    'color': '#adb5bd'
                },
                selected_style={
                    'padding': '0px',
                    'fontWeight': 'bold',
                    'backgroundColor': '#1e1e1e',
                    'color': '#e9ecef',
                    'borderBottom': '3px solid #495057'
                }
            ),
            dcc.Tab(
                label='Team Efficiency',
                value='tab-teams',
                children=team_tab,
                style={
                    'padding': '0px',
                    'fontWeight': 'bold',
                    'backgroundColor': '#1e1e1e',
                    'color': '#adb5bd'
                },
                selected_style={
                    'padding': '0px',
                    'fontWeight': 'bold',
                    'backgroundColor': '#1e1e1e',
                    'color': '#e9ecef',
                    'borderBottom': '3px solid #495057'
                }
            ),
        ],
        style={
            'backgroundColor': '#1e1e1e',
            'borderBottom': '1px solid #333',
            'padding': '0px 20px'
        }
    )
], style={"backgroundColor": "#1e1e1e", "minHeight": "100vh", "padding": "0px"})


# ==========================================
# 6. CALLBACKS
# ==========================================

@callback(
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
def update_dashboard(min_lebron, min_salary_m, max_salary_m):
    """Update all player visualizations."""
    min_salary = min_salary_m * 1000000
    max_salary = max_salary_m * 1000000
    filtered = df[(df['LEBRON'] >= min_lebron) & 
                  (df['current_year_salary'] >= min_salary) &
                  (df['current_year_salary'] <= max_salary)]
    
    # 1. Scatter: Salary vs LEBRON
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=filtered['LEBRON'],
        y=filtered['current_year_salary'] if 'current_year_salary' in filtered.columns else filtered['total_contract_value'],
        mode='markers',
        marker=dict(
            size=filtered['LEBRON WAR'].clip(lower=1) * 3 if 'LEBRON WAR' in filtered.columns else 5,
            color=filtered['value_gap'] if 'value_gap' in filtered.columns else 0,
            colorscale='RdYlGn',
            colorbar=dict(title="Value Gap", thickness=12, len=0.7),
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        text=[f"<b>{n}</b><br>Gap: {g:.1f}" for n, g in 
              zip(filtered['player_name'], filtered['value_gap'] if 'value_gap' in filtered.columns else [0]*len(filtered))],
        hovertemplate='%{text}<extra></extra>'
    ))
    fig1.update_layout(
        title='<b>Salary vs Impact</b> (Size=WAR, Color=Value Gap)',
        xaxis_title='LEBRON Total',
        yaxis_title='Salary ($)',
        height=450,
        template='plotly_dark',
        hovermode='closest'
    )
    
    # 2. Bar: Top Underpaid
    if 'value_gap' in filtered.columns and len(filtered) > 0:
        top_under = filtered.nlargest(10, 'value_gap').sort_values('value_gap', ascending=True)
        fig2 = go.Figure(go.Bar(
            x=top_under['value_gap'],
            y=top_under['player_name'],
            orientation='h',
            marker=dict(color='#28a745', opacity=0.9),
            text=[f"{v:.1f}" for v in top_under['value_gap']],
            textposition='outside'
        ))
        fig2.update_layout(
            title='<b>Top 10 Underpaid</b>',
            xaxis_title='Value Gap',
            height=450,
            template='plotly_dark',
            margin=dict(l=200),
            showlegend=False
        )
    else:
        fig2 = go.Figure().add_annotation(text="No data")
    
    # 3. Scatter: Off vs Def
    fig3 = go.Figure()
    if len(filtered) > 0:
        fig3.add_trace(go.Scatter(
            x=filtered['O-LEBRON'],
            y=filtered['D-LEBRON'],
            mode='markers',
            marker=dict(
                size=filtered['LEBRON'].abs() * 3,
                color=filtered['value_gap'] if 'value_gap' in filtered.columns else 0,
                colorscale='RdYlGn',
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=[f"<b>{n}</b>" for n in filtered['player_name']],
            hovertemplate='%{text}<extra></extra>'
        ))
    fig3.update_layout(
        title='<b>Offensive vs Defensive Impact</b>',
        xaxis_title='Offensive LEBRON',
        yaxis_title='Defensive LEBRON',
        height=450,
        template='plotly_dark',
        hovermode='closest'
    )
    
    # 4. Bar: Top Overpaid
    if 'value_gap' in filtered.columns and len(filtered) > 0:
        top_over = filtered.nsmallest(10, 'value_gap').sort_values('value_gap', ascending=False)
        fig4 = go.Figure(go.Bar(
            x=top_over['value_gap'],
            y=top_over['player_name'],
            orientation='h',
            marker=dict(color='#dc3545', opacity=0.9),
            text=[f"{v:.1f}" for v in top_over['value_gap']],
            textposition='outside'
        ))
        fig4.update_layout(
            title='<b>Top 10 Overpaid</b>',
            xaxis_title='Value Gap',
            height=450,
            template='plotly_dark',
            margin=dict(l=200),
            showlegend=False
        )
    else:
        fig4 = go.Figure().add_annotation(text="No data")
    
    # 5. Table: Underpaid
    if 'value_gap' in filtered.columns and len(filtered) > 0:
        top_under_data = filtered.nlargest(10, 'value_gap')[
            ['player_name', 'current_year_salary', 'LEBRON', 'value_gap']
        ]
        rows = []
        for _, row in top_under_data.iterrows():
            player_name = str(row['player_name']).strip() if pd.notna(row['player_name']) else 'Unknown'
            rows.append(html.Tr([
                html.Td(player_name, style={"fontWeight": "500", "width": "45%"}),
                html.Td(f"${row['current_year_salary']:,.0f}", style={"width": "30%"}),
                html.Td(f"{row['LEBRON']:.2f}", style={"width": "15%"}),
                html.Td(f"{row['value_gap']:.1f}", className="text-success fw-bold", style={"width": "10%"})
            ]))
        
        table_under = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Player"), html.Th("Salary"), html.Th("LEBRON"), html.Th("Gap")
            ], style={"fontSize": "12px"})),
            html.Tbody(rows, style={"fontSize": "12px"})
        ], striped=True, hover=True, bordered=True, size='sm')
    else:
        table_under = html.P("No data available")
    
    # 6. Table: Overpaid
    if 'value_gap' in filtered.columns and len(filtered) > 0:
        top_over_data = filtered.nsmallest(10, 'value_gap')[
            ['player_name', 'current_year_salary', 'LEBRON', 'value_gap']
        ]
        rows = []
        for _, row in top_over_data.iterrows():
            player_name = str(row['player_name']).strip() if pd.notna(row['player_name']) else 'Unknown'
            rows.append(html.Tr([
                html.Td(player_name, style={"fontWeight": "500", "width": "45%"}),
                html.Td(f"${row['current_year_salary']:,.0f}", style={"width": "30%"}),
                html.Td(f"{row['LEBRON']:.2f}", style={"width": "15%"}),
                html.Td(f"{row['value_gap']:.1f}", className="text-danger fw-bold", style={"width": "10%"})
            ]))
        
        table_over = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Player"), html.Th("Salary"), html.Th("LEBRON"), html.Th("Gap")
            ], style={"fontSize": "12px"})),
            html.Tbody(rows, style={"fontSize": "12px"})
        ], striped=True, hover=True, bordered=True, size='sm')
    else:
        table_over = html.P("No data available")
    
    # 7. Table: ALL players with color-coded gaps
    if 'value_gap' in df.columns:
        all_players_data = df.sort_values('value_gap', ascending=False)[
            ['player_name', 'current_year_salary', 'LEBRON', 'value_gap', 'archetype']
        ]
        rows = []
        for _, row in all_players_data.iterrows():
            player_name = str(row['player_name']).strip() if pd.notna(row['player_name']) else 'Unknown'
            archetype = str(row['archetype']).strip() if pd.notna(row['archetype']) else 'Unknown'
            gap_value = row['value_gap']
            
            # Multi-tier coloring
            if gap_value > 20:
                gap_color = '#28a745'
            elif gap_value > 5:
                gap_color = '#7dd97d'
            elif gap_value < -20:
                gap_color = '#dc3545'
            elif gap_value < -5:
                gap_color = '#e79595'
            else:
                gap_color = '#ffc107'

            rows.append(html.Tr([
                html.Td(player_name, style={"fontWeight": "500", "width": "25%", "fontSize": "11px"}),
                html.Td(f"${row['current_year_salary']:,.0f}", style={"width": "15%", "fontSize": "11px"}),
                html.Td(f"{row['LEBRON']:.2f}", style={"width": "10%", "fontSize": "11px"}),
                html.Td(f"{gap_value:.1f}", style={"color": gap_color, "fontWeight": "bold", "width": "10%", "fontSize": "11px"}),
                html.Td(archetype, style={"width": "40%", "fontSize": "10px"})
            ]))
        
        table_all = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Player", style={"fontSize": "11px"}),
                html.Th("Salary", style={"fontSize": "11px"}),
                html.Th("LEBRON", style={"fontSize": "11px"}),
                html.Th("Gap", style={"fontSize": "11px"}),
                html.Th("Archetype", style={"fontSize": "11px"})
            ])),
            html.Tbody(rows, style={"fontSize": "11px"})
        ], striped=True, hover=True, bordered=True, size='sm')
    else:
        table_all = html.P("No data available")
    
    return fig1, fig2, fig3, fig4, table_under, table_over, table_all


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SIEVE Dashboard Starting...")
    print("=" * 70)
    print(f"\nLoaded {len(df)} players and {len(df_teams)} teams")
    print("Open browser to: http://localhost:8050")
    print("Press CTRL+C to stop\n")
    app.run(debug=True, port=8050)
