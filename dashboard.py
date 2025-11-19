import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import os


print("Loading and preparing data...")

lebron_file = 'LEBRON.csv'
contracts_file = 'basketball_reference_contracts.csv'  # Changed from hoopshype_contracts.csv

if not os.path.exists(lebron_file) or not os.path.exists(contracts_file):
    print(f"ERROR: Missing {lebron_file} or {contracts_file}")
    exit(1)

# Load
df_lebron = pd.read_csv(lebron_file)
df_contracts = pd.read_csv(contracts_file)

# Rename for consistency
df_lebron = df_lebron.rename(columns={'Player': 'player_name'})

# Convert to numeric
for col in ['LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON']:
    if col in df_lebron.columns:
        df_lebron[col] = pd.to_numeric(df_lebron[col], errors='coerce')

# Create combined archetype label from existing columns
if 'Offensive Archetype' in df_lebron.columns and 'Defensive Role' in df_lebron.columns:
    df_lebron['archetype'] = (
        df_lebron['Offensive Archetype'].fillna('Unknown').astype(str) + 
        ' / ' + 
        df_lebron['Defensive Role'].fillna('Unknown').astype(str)
    )
else:
    df_lebron['archetype'] = 'Unknown'

# Merge - use inner join to only get players with both archetype and salary data
df = pd.merge(df_lebron, df_contracts, on='player_name', how='inner')
df = df.drop_duplicates(subset=['player_name'], keep='first')



# Handle NaN in current_year_salary - fill with average_annual_value if available
if 'current_year_salary' in df.columns:
    nan_mask = df['current_year_salary'].isna()
    if 'average_annual_value' in df.columns:
        df.loc[nan_mask, 'current_year_salary'] = df.loc[nan_mask, 'average_annual_value']

# Calculate value gap: normalized salary vs normalized impact
if 'current_year_salary' in df.columns and 'LEBRON' in df.columns:
    # Remove NaN values for normalization
    valid_salary = df['current_year_salary'].dropna()
    valid_lebron = df['LEBRON'].dropna()
    
    if len(valid_salary) > 0 and len(valid_lebron) > 0:
        # Normalize salary 0-100
        salary_min = valid_salary.min()
        salary_max = valid_salary.max()
        df['salary_norm'] = 100 * (df['current_year_salary'] - salary_min) / (salary_max - salary_min)
        
        # Normalize LEBRON 0-100
        lebron_min = valid_lebron.min()
        lebron_max = valid_lebron.max()
        df['impact_norm'] = 100 * (df['LEBRON'] - lebron_min) / (lebron_max - lebron_min)
        
        # Value gap: positive = underpaid, negative = overpaid
        df['value_gap'] = df['impact_norm'] - df['salary_norm']
    else:
        df['value_gap'] = 0



print(f"Loaded {len(df)} players")
print(f"Archetypes: {df['archetype'].nunique()}")



#creating the dashboard
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Sieve", className="display-4 text-center mt-4 mb-1",
                   style={"fontWeight": "700"}),
            html.H4("NBA Archetypes: Salary, Impact & Value", className="text-center mb-4 text-muted")
        ])
    ]),
    
    # Filters
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
    ], className="mb-5"),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='chart-salary-impact')
        ], md=6),
        dbc.Col([
            dcc.Graph(id='chart-underpaid')
        ], md=6)
    ], className="mb-4"),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='chart-off-def')
        ], md=6),
        dbc.Col([
            dcc.Graph(id='chart-overpaid')
        ], md=6)
    ], className="mb-4"),
    
    # Tables Row 1: Top Underpaid/Overpaid
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
    
    # Tables Row 2: All Filtered Players
    dbc.Row([
        dbc.Col([
            html.H5(f"All {len(df)} Players (Sorted by Value Gap)", className="mt-3 mb-2"),
            html.Div(id='table-all-players', style={"maxHeight": "800px", "overflowY": "auto"})
        ])
    ], className="mb-5")
    
], fluid=True, style={"backgroundColor": "#1e1e1e", "minHeight": "100vh", "padding": "20px"})

@callback(
    [Output('chart-salary-impact', 'figure'),
     Output('chart-underpaid', 'figure'),
     Output('chart-off-def', 'figure'),
     Output('chart-overpaid', 'figure'),
     Output('table-underpaid', 'children'),
     Output('table-overpaid', 'children'),
     Output('table-all-players', 'children')],
    [Input('min-lebron', 'value'),
     Input('max-salary', 'value')]
)
def update_dashboard(min_lebron, max_salary_m):
    """Update all visualizations."""
    max_salary = max_salary_m * 1000000
    filtered = df[(df['LEBRON'] >= min_lebron) & 
                  ((df['current_year_salary'] <= max_salary) if 'current_year_salary' in df.columns else True)]
    
    # 1. Scatter: Salary vs LEBRON (colored by value_gap)
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
    
    # 2. Bar: Top Underpaid (GREEN)
    if 'value_gap' in filtered.columns:
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
            title='<b>Top 10 Underpaid</b> ',
            xaxis_title='Value Gap',
            height=450,
            template='plotly_dark',
            margin=dict(l=200),
            showlegend=False
        )
    else:
        fig2 = go.Figure().add_annotation(text="No value_gap data")
    
    # 3. Scatter: Offensive vs Defensive
    fig3 = go.Figure()
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
    
    # 4. Bar: Top Overpaid (RED)
    if 'value_gap' in filtered.columns:
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
            title='<b>Top 10 Overpaid</b> ',
            xaxis_title='Value Gap',
            height=450,
            template='plotly_dark',
            margin=dict(l=200),
            showlegend=False
        )
    else:
        fig4 = go.Figure().add_annotation(text="No value_gap data")
    
    # 5. Table: Underpaid
    if 'value_gap' in filtered.columns:
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
                html.Th("Player"),
                html.Th("Salary"),
                html.Th("LEBRON"),
                html.Th("Gap")
            ], style={"fontSize": "12px"})),
            html.Tbody(rows, style={"fontSize": "12px"})
        ], striped=True, hover=True, bordered=True, size='sm')
    else:
        table_under = html.P("No data available")
    
    # 6. Table: Overpaid
    if 'value_gap' in filtered.columns:
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
                html.Th("Player"),
                html.Th("Salary"),
                html.Th("LEBRON"),
                html.Th("Gap")
            ], style={"fontSize": "12px"})),
            html.Tbody(rows, style={"fontSize": "12px"})
        ], striped=True, hover=True, bordered=True, size='sm')
    else:
        table_over = html.P("No data available")
    
    # 7. Table: ALL players in entire dataset (NOT filtered by sliders) - sorted by value_gap
    if 'value_gap' in df.columns:
        all_players_data = df.sort_values('value_gap', ascending=False)[
            ['player_name', 'current_year_salary', 'LEBRON', 'value_gap', 'archetype']
        ]
        rows = []
        for _, row in all_players_data.iterrows():
            player_name = str(row['player_name']).strip() if pd.notna(row['player_name']) else 'Unknown'
            archetype = str(row['archetype']).strip() if pd.notna(row['archetype']) else 'Unknown'
            gap_value = row['value_gap']
            
            if gap_value > 20:
                gap_color = '#28a745' #Green
            elif gap_value > 5:
                gap_color = '#7dd97d'  #Lighter green
            elif gap_value < -20:
                gap_color = '#dc3545' #Red
            elif gap_value < -5:  
                gap_color = '#e79595'#Lighter red
            else:
                gap_color = '#ffc107'#yellow

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


#running
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SIEVE Dashboard Starting...")
    print("=" * 70)
    print("\nOpen browser to: http://localhost:8050")
    print("Press CTRL+C to stop\n")
    app.run(debug=True, port=8050)