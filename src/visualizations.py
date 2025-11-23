"""
Visualization module for Sieve NBA Analytics.
This module contains functions to generate Plotly figures and Dash tables.
It separates the view logic from the main application controller.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dash_table


# TEAM VISUALIZATIONS

def create_efficiency_quadrant(df_teams):
    """
    Creates the Efficiency Quadrant Chart (Wins vs. Payroll).
    
    This scatter plot visualizes team performance relative to their spending.
    It includes a background contour plot representing the 'Efficiency Index'
    (Z-score of Wins minus Z-score of Payroll) to clearly demarcate 
    efficient vs. inefficient zones.

    Visual Elements:
    - X-Axis: Total Payroll (USD).
    - Y-Axis: Total Wins.
    - Markers: Official NBA Team Logos.
    - Background: Red-Yellow-Green gradient indicating efficiency relative to league average.

    Args:
        df_teams (pd.DataFrame): DataFrame containing team metrics (WINS, Total_Payroll, Logo_URL).

    Returns:
        go.Figure: A Plotly Graph Object containing the quadrant chart.
    """
    if df_teams.empty:
        fig = go.Figure().add_annotation(text="No Team Data Available")
        fig.update_layout(height=600, template='plotly_dark', paper_bgcolor='#0f1623')
        return fig

    avg_payroll = df_teams['Total_Payroll'].mean()
    avg_wins = df_teams['WINS'].mean()
    
    # Calculate tighter bounds to "zoom in" on the relevant data area
    x_min, x_max = df_teams['Total_Payroll'].min(), df_teams['Total_Payroll'].max()
    y_min, y_max = df_teams['WINS'].min(), df_teams['WINS'].max()
    x_padding = (x_max - x_min) * 0.15  # Increased padding slightly
    y_padding = (y_max - y_min) * 0.15
    
    view_x_min = x_min - x_padding
    view_x_max = x_max + x_padding
    view_y_min = y_min - y_padding
    view_y_max = y_max + y_padding
    
    # Initialize the figure
    fig_quadrant = go.Figure()

    # 1. Add Background Gradient (Relative to League Average)
    # We calculate Z-scores for a meshgrid to create a contour plot.
    # This ensures the gradient is centered on the "Average Team" (0,0 in Z-space).
    
    mean_pay = df_teams['Total_Payroll'].mean()
    std_pay = df_teams['Total_Payroll'].std()
    mean_wins = df_teams['WINS'].mean()
    std_wins = df_teams['WINS'].std()
    
    # Define the grid range to slightly EXCEED the view range to ensure full coverage
    x_range = np.linspace(view_x_min * 0.95, view_x_max * 1.05, 100)
    y_range = np.linspace(view_y_min * 0.95, view_y_max * 1.05, 100)
    
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate Z-scores for every point on the grid
    # Z_score = (Value - Mean) / StdDev
    Z_pay = (X - mean_pay) / std_pay
    Z_wins = (Y - mean_wins) / std_wins
    
    # Efficiency = (2.0 * Win_Z_Score) - Payroll_Z_Score
    # We weight Wins (2.0) higher than Payroll (1.0) to prioritize on-court success.
    Z = (2.0 * Z_wins) - Z_pay
    
    # Add the contour plot as the background layer
    fig_quadrant.add_trace(go.Contour(
        z=Z, x=x_range, y=y_range,
        colorscale='RdYlGn',  # Red to Green colormap
        opacity=0.4,          # Semi-transparent to let grid lines show
        showscale=False,      # Hide the color bar
        hoverinfo='skip',     # Disable hover for the background
        contours=dict(
            coloring='heatmap',
            start=-3,
            end=3,
            size=0.2
        )
    ))

    # 2. Add Invisible Scatter for Hover Data
    # Since we are using images for markers, we need a separate invisible scatter trace
    # to handle the hover tooltips.
    fig_quadrant.add_trace(go.Scatter(
        x=df_teams['Total_Payroll'],
        y=df_teams['WINS'],
        mode='markers',
        text=df_teams['Abbrev'],
        customdata=np.stack((df_teams['Total_WAR'], df_teams['Efficiency_Index']), axis=-1),
        hovertemplate='<b>%{text}</b><br>Wins: %{y}<br>Payroll: $%{x:,.0f}<br>WAR: %{customdata[0]:.1f}<br>Eff Index: %{customdata[1]:.2f}<extra></extra>',
        marker=dict(opacity=0) # Fully transparent markers
    ))

    # 3. Add Team Logos as Images
    logo_images = []
    if 'Logo_URL' in df_teams.columns:
        for _, row in df_teams.iterrows():
            if pd.notna(row['Logo_URL']):
                logo_images.append(dict(
                    source=row['Logo_URL'],
                    xref="x", yref="y",
                    x=row['Total_Payroll'], y=row['WINS'],
                    sizex=35000000, sizey=8, # Adjust size to be visible but not overwhelming
                    xanchor="center", yanchor="middle",
                    layer="above"
                ))

    # Configure the layout
    fig_quadrant.update_layout(
        # title='<b>Efficiency Quadrant: Wins vs. Payroll</b><br><sub style="color:#adb5bd">Green = Outperforming Budget | Red = Underperforming Budget</sub>',
        xaxis_title='<b>Total Payroll ($)</b>',
        yaxis_title='<b>Wins</b>',
        height=650,
        margin=dict(l=80, r=40, t=20, b=70),
        paper_bgcolor='#0f1623', # Deep Navy background
        plot_bgcolor='#1a202c',  # Slightly lighter plot area
        font=dict(size=12),
        images=logo_images,
        xaxis=dict(showgrid=True, gridcolor='#2c3e50', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#2c3e50', zeroline=False),
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb")
        )
    )
    
    # 4. Add Quadrant Lines (League Averages)
    # These dashed lines indicate the average payroll and average wins.
    fig_quadrant.add_vline(x=avg_payroll, line_dash="dash", line_color="rgba(255,255,255,0.6)", line_width=1.5,
                          annotation_text="Avg Payroll", annotation_position="top right", annotation_font_color="#adb5bd")
    fig_quadrant.add_hline(y=avg_wins, line_dash="dash", line_color="rgba(255,255,255,0.6)", line_width=1.5,
                          annotation_text="Avg Wins", annotation_position="bottom right", annotation_font_color="#adb5bd")
    
    # Add Text Labels for Context (Quadrant Names)
    fig_quadrant.add_annotation(x=df_teams['Total_Payroll'].min(), y=df_teams['WINS'].max(), 
                               text="<b>ELITE</b><br>(High Wins / Low Pay)", showarrow=False, 
                               font=dict(color="#06d6a0", size=14, weight="bold"), xanchor="left", yanchor="top",
                               bgcolor="rgba(15, 22, 35, 0.7)", borderpad=4)
                               
    fig_quadrant.add_annotation(x=df_teams['Total_Payroll'].max(), y=df_teams['WINS'].max(), 
                               text="<b>CONTENDERS</b><br>(High Wins / High Pay)", showarrow=False, 
                               font=dict(color="#ffd166", size=14, weight="bold"), xanchor="right", yanchor="top",
                               bgcolor="rgba(15, 22, 35, 0.7)", borderpad=4)
                               
    fig_quadrant.add_annotation(x=df_teams['Total_Payroll'].min(), y=df_teams['WINS'].min(), 
                               text="<b>REBUILDING</b><br>(Low Wins / Low Pay)", showarrow=False, 
                               font=dict(color="#e4e6eb", size=12), xanchor="left", yanchor="bottom",
                               bgcolor="rgba(15, 22, 35, 0.7)", borderpad=4)
                               
    fig_quadrant.add_annotation(x=df_teams['Total_Payroll'].max(), y=df_teams['WINS'].min(), 
                               text="<b>DISASTER</b><br>(Low Wins / High Pay)", showarrow=False, 
                               font=dict(color="#ef476f", size=14, weight="bold"), xanchor="right", yanchor="bottom",
                               bgcolor="rgba(15, 22, 35, 0.7)", borderpad=4)
    
    return fig_quadrant


def create_team_grid(df_teams):
    """
    Creates the Team Efficiency Grid.
    
    This visualization displays all 30 teams as equal-sized tiles in a 6x5 grid:
    - Sorted by Efficiency Index (Best to Worst, top-left to bottom-right)
    - Color coded by efficiency
    - Team logos prominently displayed

    Args:
        df_teams (pd.DataFrame): DataFrame containing team metrics.

    Returns:
        go.Figure: A Plotly Graph Object containing the grid.
    """
    if df_teams.empty:
        fig = go.Figure().add_annotation(text="No Team Data Available")
        fig.update_layout(height=600, template='plotly_dark', paper_bgcolor='#0f1623')
        return fig

    # Sort teams by Efficiency Index descending (Best to Worst)
    df_grid = df_teams.sort_values('Efficiency_Index', ascending=False).reset_index(drop=True)
    
    # Create grid coordinates (6 columns, 5 rows)
    cols = 6
    df_grid['grid_x'] = df_grid.index % cols
    df_grid['grid_y'] = df_grid.index // cols
    # Invert Y so top rank is at top-left (visual convention)
    df_grid['grid_y'] = df_grid['grid_y'].max() - df_grid['grid_y']
    
    fig_grid = go.Figure()
    
    # Add colored tiles (Heatmap-like effect using Scatter markers)
    fig_grid.add_trace(go.Scatter(
        x=df_grid['grid_x'],
        y=df_grid['grid_y'],
        mode='markers',
        marker=dict(
            symbol='square',
            size=65,  # Large squares to form tiles
            color=df_grid['Efficiency_Index'],
            colorscale='RdYlGn',
            cmin=-3, cmax=3,  # Fixed scale for consistency with Quadrant chart
            colorbar=dict(
                title='Efficiency',
                tickmode='linear',
                tick0=-3,
                dtick=1,
                thickness=15,
                len=0.7,
                bgcolor='#1a202c',
                bordercolor='#2c3e50',
                borderwidth=1,
                tickfont=dict(color='#e4e6eb', size=11)
            ),
            line=dict(width=2, color='#1a2332'),
            opacity=0.9
        ),
        text=df_grid['Abbrev'],
        customdata=np.stack((df_grid['Efficiency_Index'], df_grid['Total_Payroll'], df_grid['WINS']), axis=-1),
        hovertemplate='<b>%{text}</b><br>Eff Index: %{customdata[0]:.2f}<br>Payroll: $%{customdata[1]:,.0f}<br>Wins: %{customdata[2]}<extra></extra>'
    ))
    
    # Add Team Logos on top of the tiles
    grid_images = []
    if 'Logo_URL' in df_grid.columns:
        for _, row in df_grid.iterrows():
            if pd.notna(row['Logo_URL']):
                grid_images.append(dict(
                    source=row['Logo_URL'],
                    xref="x", yref="y",
                    x=row['grid_x'], y=row['grid_y'],
                    sizex=0.7, sizey=0.7,
                    xanchor="center", yanchor="middle",
                    layer="above"
                ))
    
    fig_grid.update_layout(
        # title='<b>Team Efficiency Rankings</b><br><sub style="color:#adb5bd">Sorted by Efficiency • Green = Good Value • Red = Overpaying</sub>',
        height=650,
        margin=dict(l=80, r=40, t=20, b=70),
        paper_bgcolor='#0f1623',
        plot_bgcolor='#1a202c',
        font=dict(size=12),
        # Hide axes for a clean grid look
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 5.5], fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5], fixedrange=True),
        images=grid_images,
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb", size=12)
        )
    )
    return fig_grid



# PLAYER VISUALIZATIONS
def create_salary_impact_scatter(filtered):
    """
    Creates the Salary vs. Impact Scatter Plot.
    
    This chart helps identify outliers: players who provide high impact for low salary (top-left)
    vs. players who provide low impact for high salary (bottom-right).

    Args:
        filtered (pd.DataFrame): Filtered DataFrame of players.

    Returns:
        go.Figure: A Plotly scatter plot.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered['LEBRON'],
        y=filtered['current_year_salary'] if 'current_year_salary' in filtered.columns else filtered['total_contract_value'],
        mode='markers',
        marker=dict(
            # Size markers by WAR (Wins Above Replacement) to show total contribution
            size=filtered['LEBRON WAR'].clip(lower=1) * 3.5 if 'LEBRON WAR' in filtered.columns else 6,
            # Color markers by Value Gap to highlight efficiency
            color=filtered['value_gap'] if 'value_gap' in filtered.columns else 0,
            colorscale=[[0, '#ef476f'], [0.5, '#ffd166'], [1, '#06d6a0']],
            colorbar=dict(
                title=dict(text="<b>Value<br>Gap</b>", font=dict(size=11)),
                thickness=15,
                len=0.6,
                tickfont=dict(size=10)
            ),
            line=dict(width=1.5, color='rgba(255,255,255,0.4)'),
            opacity=0.85
        ),
        text=[f"<b>{n}</b><br>Gap: {g:.1f}" for n, g in 
              zip(filtered['player_name'], filtered['value_gap'] if 'value_gap' in filtered.columns else [0]*len(filtered))],
        hovertemplate='%{text}<extra></extra>'
    ))
    fig.update_layout(
        # title='<b style="font-size:16px">Salary vs Impact</b><br><sub style="color:#adb5bd">Size = WAR | Color = Value Gap</sub>',
        xaxis_title='<b>LEBRON Total</b>',
        yaxis_title='<b>Salary ($)</b>',
        height=550,
        template='plotly_dark',
        hovermode='closest',
        paper_bgcolor='#0f1623',
        plot_bgcolor='#1a202c',
        margin=dict(l=70, r=100, t=20, b=60),
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb")
        )
    )
    return fig


def create_underpaid_bar(filtered):
    """
    Creates a horizontal bar chart of the Top 20 Underpaid Players.
    
    Args:
        filtered (pd.DataFrame): Filtered DataFrame of players.

    Returns:
        go.Figure: A Plotly bar chart.
    """
    if 'value_gap' in filtered.columns and len(filtered) > 0:
        # Filter for positive value gaps only
        underpaid_only = filtered[filtered['value_gap'] > 0]
        if len(underpaid_only) > 0:
            # Get top 20
            top_under = underpaid_only.nlargest(20, 'value_gap').sort_values('value_gap', ascending=True)
            fig = go.Figure(go.Bar(
                x=top_under['value_gap'],
                y=top_under['player_name'],
                orientation='h',
                marker=dict(
                    color='#06d6a0',  # Electric Green
                    opacity=0.9,
                    line=dict(color='#04a077', width=1)
                ),
                text=[f"{v:.1f}" for v in top_under['value_gap']],
                textposition='outside',
                textfont=dict(size=11, color='white')
            ))
            fig.update_layout(
                # title='<b style="font-size:16px">Top 20 Underpaid Players</b>',
                xaxis_title='<b>Value Gap</b>',
                height=550,
                template='plotly_dark',
                margin=dict(l=200, r=40, t=20, b=60),
                showlegend=False,
                paper_bgcolor='#0f1623',
                plot_bgcolor='#1a202c',
                hoverlabel=dict(
                    bgcolor="#1a2332",
                    bordercolor="#ff6b35",
                    font=dict(color="#e4e6eb")
                )
            )
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            return fig
            
    # Fallback if no data meets criteria
    fig = go.Figure().add_annotation(text="No underpaid players in current filter")
    fig.update_layout(height=500, template='plotly_dark', paper_bgcolor='#0f1623')
    return fig


def create_age_impact_scatter(filtered):
    """
    Creates the Age vs. Impact Scatter Plot (Development Curve).
    
    This chart visualizes how player impact correlates with age, helping to identify
    peak performance windows and aging curves.

    Args:
        filtered (pd.DataFrame): Filtered DataFrame of players.

    Returns:
        go.Figure: A Plotly scatter plot.
    """
    fig = go.Figure()
    if len(filtered) > 0 and 'Age' in filtered.columns:
        fig.add_trace(go.Scatter(
            x=filtered['Age'],
            y=filtered['value_gap'],
            mode='markers',
            marker=dict(
                # Size by Salary to show if expensive players are performing
                size=filtered['current_year_salary'].fillna(0) / 2_000_000 if 'current_year_salary' in filtered.columns else 10,
                color=filtered['value_gap'] if 'value_gap' in filtered.columns else 0,
                colorscale=[[0, '#ef476f'], [0.5, '#ffd166'], [1, '#06d6a0']],
                line=dict(width=1.5, color='rgba(255,255,255,0.4)'),
                opacity=0.85,
                sizemin=4
            ),
            text=[f"<b>{n}</b><br>Age: {a}<br>Impact: {i:.2f}<br>Gap: {g:.1f}" 
                  for n, a, i, g in zip(
                      filtered['player_name'], 
                      filtered['Age'], 
                      filtered['value_gap'],
                      filtered['value_gap'] if 'value_gap' in filtered.columns else [0]*len(filtered)
                  )],
            hovertemplate='%{text}<extra></extra>'
        ))
    fig.update_layout(
        # title='<b style="font-size:16px">Age vs Impact Curve</b><br><sub style="color:#adb5bd">Size = Salary | Color = Value Gap</sub>',
        xaxis_title='<b>Player Age</b>',
        yaxis_title='<b>Value Gap Impact</b>',
        height=550,
        template='plotly_dark',
        hovermode='closest',
        paper_bgcolor='#0f1623',
        plot_bgcolor='#1a202c',
        margin=dict(l=70, r=40, t=20, b=60),
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb")
        )
    )
    # Add reference band for typical "Peak Years" (26-30)
    fig.add_vrect(x0=26, x1=30, fillcolor="rgba(255,107,53,0.1)", 
                   line_width=0, annotation_text="Peak Years", 
                   annotation_position="top left")
    return fig


def create_overpaid_bar(filtered):
    """
    Creates a horizontal bar chart of the Top 20 Overpaid Players.
    
    Args:
        filtered (pd.DataFrame): Filtered DataFrame of players.

    Returns:
        go.Figure: A Plotly bar chart.
    """
    if 'value_gap' in filtered.columns and len(filtered) > 0:
        overpaid_only = filtered[filtered['value_gap'] < 0]
        if len(overpaid_only) > 0:
            top_over = overpaid_only.nsmallest(20, 'value_gap').sort_values('value_gap', ascending=False)
            fig = go.Figure(go.Bar(
                x=top_over['value_gap'],
                y=top_over['player_name'],
                orientation='h',
                marker=dict(
                    color='#ef476f',  # Hot Pink
                    opacity=0.9,
                    line=dict(color='#c92a4e', width=1)
                ),
                text=[f"{v:.1f}" for v in top_over['value_gap']],
                textposition='outside',
                textfont=dict(size=11, color='white')
            ))
            fig.update_layout(
                # title='<b style="font-size:16px">Top 20 Overpaid Players</b>',
                xaxis_title='<b>Value Gap</b>',
                height=550,
                template='plotly_dark',
                margin=dict(l=200, r=40, t=20, b=60),
                showlegend=False,
                paper_bgcolor='#0f1623',
                plot_bgcolor='#1a202c',
                hoverlabel=dict(
                    bgcolor="#1a2332",
                    bordercolor="#ff6b35",
                    font=dict(color="#e4e6eb")
                )
            )
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            return fig
            
    # Fallback
    fig = go.Figure().add_annotation(text="No overpaid players in current filter")
    fig.update_layout(height=500, template='plotly_dark', paper_bgcolor='#0f1623')
    return fig


# TABLES
def create_player_table(filtered, table_type='underpaid'):
    """
    Creates a Dash DataTable for displaying player lists.
    
    Args:
        filtered (pd.DataFrame): Filtered DataFrame of players.
        table_type (str): 'underpaid' or 'overpaid'. Determines sorting and coloring.

    Returns:
        dash_table.DataTable: A styled table component.
    """
    if 'value_gap' not in filtered.columns or len(filtered) == 0:
        return html.P("No data available")

    if table_type == 'underpaid':
        data = filtered[filtered['value_gap'] > 0]
        if len(data) == 0: return html.P("No underpaid players in filter")
        top_data = data.nlargest(20, 'value_gap')[['player_name', 'current_year_salary', 'LEBRON', 'value_gap']].copy()
        text_color = "#06d6a0"
    elif table_type == 'overpaid':
        data = filtered[filtered['value_gap'] < 0]
        if len(data) == 0: return html.P("No overpaid players in filter")
        top_data = data.nsmallest(20, 'value_gap')[['player_name', 'current_year_salary', 'LEBRON', 'value_gap']].copy()
        text_color = "#ef476f"
    else:
        return html.P("Invalid table type")

    # Clean up player names
    top_data['player_name'] = top_data['player_name'].apply(
        lambda x: str(x).strip() if pd.notna(x) else 'Unknown'
    )
    
    return dash_table.DataTable(
        data=top_data.to_dict('records'),
        columns=[
            {'name': 'Player', 'id': 'player_name', 'type': 'text'},
            {'name': 'Salary', 'id': 'current_year_salary', 'type': 'numeric', 
             'format': {'specifier': '$,.0f'}},
            {'name': 'LEBRON', 'id': 'LEBRON', 'type': 'numeric', 
             'format': {'specifier': '.2f'}},
            {'name': 'Gap', 'id': 'value_gap', 'type': 'numeric', 
             'format': {'specifier': '.1f'}}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={
            'backgroundColor': '#1a2332',
            'color': '#e4e6eb',
            'textAlign': 'left',
            'padding': '6px 12px',
            'fontSize': '12px',
            'border': 'none',
            'borderBottom': '1px solid #2c3e50'
        },
        style_header={
            'backgroundColor': '#151b26',
            'fontWeight': 'bold',
            'textAlign': 'left',
            'color': '#e4e6eb',
            'borderBottom': '2px solid #ff6b35',
            'borderTop': 'none',
            'borderLeft': 'none',
            'borderRight': 'none',
            'fontSize': '12px',
            'padding': '6px 12px'
        },
        style_data_conditional=[
            # Player name styling
            {
                'if': {'column_id': 'player_name'},
                'fontWeight': '500'
            },
            # Value gap color
            {
                'if': {'column_id': 'value_gap'},
                'color': text_color,
                'fontWeight': 'bold'
            }
        ],
        page_action='none'
    )


def create_all_players_table(df):
    """
    Creates the comprehensive 'All Players' table with sorting capabilities.
    
    This table lists every player in the dataset, initially sorted by Value Gap.
    Users can click column headers to sort by any column.
    It uses color-coded text to highlight extreme values.

    Args:
        df (pd.DataFrame): The full player DataFrame.

    Returns:
        dash_table.DataTable: A sortable styled table component.
    """
    if 'value_gap' not in df.columns:
        return html.P("No data available")
        
    # Prepare the data
    all_players_data = df.sort_values('value_gap', ascending=False)[
        ['player_name', 'archetype', 'current_year_salary', 'LEBRON', 'value_gap']
    ].copy()
    
    # Clean up player names and archetypes
    all_players_data['player_name'] = all_players_data['player_name'].apply(
        lambda x: str(x).strip() if pd.notna(x) else 'Unknown'
    )
    all_players_data['archetype'] = all_players_data['archetype'].apply(
        lambda x: str(x).strip() if pd.notna(x) else 'Unknown'
    )
    
    # Create style conditions based on actual values (not row indices)
    # This ensures colors stay correct even when table is sorted differently
    style_conditions = [
        # Value Gap color coding based on actual values
        {
            'if': {
                'filter_query': '{value_gap} > 20',
                'column_id': 'value_gap'
            },
            'color': '#06d6a0',  # Electric Green (Elite Value)
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{value_gap} > 5 && {value_gap} <= 20',
                'column_id': 'value_gap'
            },
            'color': '#4cd9b0',  # Good Value
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{value_gap} >= -5 && {value_gap} <= 5',
                'column_id': 'value_gap'
            },
            'color': '#ffd166',  # Warm Yellow (Neutral)
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{value_gap} >= -20 && {value_gap} < -5',
                'column_id': 'value_gap'
            },
            'color': '#ff8fa3',  # Bad Value
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{value_gap} < -20',
                'column_id': 'value_gap'
            },
            'color': '#ef476f',  # Hot Pink (Terrible Value)
            'fontWeight': 'bold'
        },
        # Player name styling
        {
            'if': {'column_id': 'player_name'},
            'fontWeight': '500'
        },
        # Archetype styling
        {
            'if': {'column_id': 'archetype'},
            'fontSize': '12px',
            'color': '#adb5bd'
        }
    ]
    
    return dash_table.DataTable(
        data=all_players_data.to_dict('records'),
        columns=[
            {'name': 'Player Name', 'id': 'player_name', 'type': 'text'},
            {'name': 'Archetype', 'id': 'archetype', 'type': 'text'},
            {'name': 'Salary', 'id': 'current_year_salary', 'type': 'numeric', 
             'format': {'specifier': '$,.0f'}},
            {'name': 'LEBRON', 'id': 'LEBRON', 'type': 'numeric', 
             'format': {'specifier': '.2f'}},
            {'name': 'Value Gap', 'id': 'value_gap', 'type': 'numeric', 
             'format': {'specifier': '.1f'}}
        ],
        sort_action='native',
        sort_mode='single',
        sort_by=[{'column_id': 'value_gap', 'direction': 'desc'}],
        style_table={'overflowX': 'auto'},
        style_cell={
            'backgroundColor': '#1a2332',
            'color': '#e4e6eb',
            'textAlign': 'left',
            'padding': '6px 12px',
            'fontSize': '13px',
            'border': 'none',
            'borderBottom': '1px solid #2c3e50'
        },
        style_header={
            'backgroundColor': '#151b26',
            'fontWeight': 'bold',
            'textAlign': 'left',
            'color': '#e4e6eb',
            'borderBottom': '2px solid #ff6b35',
            'borderTop': 'none',
            'borderLeft': 'none',
            'borderRight': 'none',
            'fontSize': '13px',
            'padding': '6px 12px'
        },
        style_data_conditional=style_conditions,
        page_action='none'
    )


def create_team_radar_chart(radar_data_team1, radar_data_team2, team1_name, team2_name):
    """
    Creates a Radar Chart (Spider Web) comparing two teams' strengths/weaknesses.
    
    Args:
        radar_data_team1 (dict): Dictionary of metrics and percentiles (0-100) for team 1.
        radar_data_team2 (dict): Dictionary of metrics and percentiles (0-100) for team 2.
        team1_name (str): Name/abbreviation of the first team.
        team2_name (str): Name/abbreviation of the second team.
        
    Returns:
        go.Figure: Plotly radar chart comparing both teams.
    """
    if not radar_data_team1 or not radar_data_team2:
        fig = go.Figure().add_annotation(
            text="No Advanced Data Available<br><sub>Select teams to compare</sub>",
            font=dict(size=14, color='#adb5bd')
        )
        fig.update_layout(height=500, template='plotly_dark', paper_bgcolor='#0f1623')
        return fig
        
    categories = list(radar_data_team1.keys())
    values_team1 = list(radar_data_team1.values())
    values_team2 = list(radar_data_team2.values())
    
    # Close the loop for radar chart
    categories.append(categories[0])
    values_team1.append(values_team1[0])
    values_team2.append(values_team2[0])
    
    fig = go.Figure()
    
    # Add Team 1 (vibrant orange/red)
    fig.add_trace(go.Scatterpolar(
        r=values_team1,
        theta=categories,
        fill='toself',
        name=team1_name,
        line=dict(color='#ff6b35', width=3),
        fillcolor='rgba(255, 107, 53, 0.25)',
        hovertemplate='<b>%{theta}</b><br>' + team1_name + ': %{r:.1f}th percentile<extra></extra>',
        marker=dict(size=8, color='#ff6b35')
    ))
    
    # Add Team 2 (vibrant blue)
    fig.add_trace(go.Scatterpolar(
        r=values_team2,
        theta=categories,
        fill='toself',
        name=team2_name,
        line=dict(color='#2D96C7', width=3),
        fillcolor='rgba(45, 150, 199, 0.25)',
        hovertemplate='<b>%{theta}</b><br>' + team2_name + ': %{r:.1f}th percentile<extra></extra>',
        marker=dict(size=8, color='#2D96C7')
    ))
    
    # Add reference circles for context (25th, 50th, 75th percentiles)
    fig.add_trace(go.Scatterpolar(
        r=[25] * len(categories),
        theta=categories,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.1)', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(categories),
        theta=categories,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.15)', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[75] * len(categories),
        theta=categories,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.1)', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickmode='array',
                tickvals=[25, 50, 75],
                ticktext=['25th', '50th', '75th'],
                tickfont=dict(size=10, color='#6c757d'),
                gridcolor='#2c3e50',
                linecolor='#2c3e50',
                gridwidth=1.5
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color='#e4e6eb', family='Inter, sans-serif'),
                gridcolor='#3a4555',
                linecolor='#3a4555',
                gridwidth=1.5
            ),
            bgcolor='#151b26'
        ),
        # title=dict(
        #     text=f'<b style="color:#e4e6eb">Team Comparison Radar</b><br>' +
        #          f'<sub style="color:#ff6b35">{team1_name}</sub> vs <sub style="color:#2D96C7">{team2_name}</sub><br>' +
        #          f'<sub style="color:#6c757d; font-size:11px">Percentile Rankings • League-Wide Comparison</sub>',
        #     y=0.97,
        #     x=0.5,
        #     xanchor='center',
        #     font=dict(size=16)
        # ),
        height=600,
        template='plotly_dark',
        paper_bgcolor='#0f1623',
        plot_bgcolor='#151b26',
        margin=dict(l=80, r=80, t=30, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(26, 35, 50, 0.8)',
            bordercolor='#3a4555',
            borderwidth=1,
            font=dict(size=12, color='#e4e6eb')
        ),
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb", size=12)
        )
    )
    
    return fig

