"""
Visualization module for Sieve NBA Analytics.
This module contains functions to generate Plotly figures and Dash tables.
It separates the view logic from the main application controller.
"""

import plotly.graph_objects as go

import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dash_table


# TEAM VISUALIZATIONS

# NBA Team Colors (Primary)
NBA_TEAM_COLORS = {
    'ATL': '#C8102E', 'BOS': '#007A33', 'BKN': '#FFFFFF', 'CHA': '#1D1160', 'CHI': '#CE1141',
    'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240', 'DET': '#C8102E', 'GSW': '#1D428A',
    'HOU': '#CE1141', 'IND': '#002D62', 'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9',
    'MIA': '#98002E', 'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#006BB6',
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#1D1160', 'POR': '#E03A3E',
    'SAC': '#5A2D81', 'SAS': '#C4CED4', 'TOR': '#CE1141', 'UTA': '#002B5C', 'WAS': '#002B5C'
}
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
                    sizex=20000000, sizey=5, # Smaller logos for cleaner look
                    xanchor="center", yanchor="middle",
                    layer="above"
                ))

    # Configure the layout
    fig_quadrant.update_layout(
        # title='<b>Efficiency Quadrant: Wins vs. Payroll</b><br><sub style="color:#adb5bd">Green = Outperforming Budget | Red = Underperforming Budget</sub>',
        xaxis_title='<b>Total Payroll ($)</b>',
        yaxis_title='<b>Wins</b>',
        height=650,
        margin=dict(l=50, r=20, t=20, b=50),  # Reduced margins for mobile
        paper_bgcolor='#0f1623', # Deep Navy background
        plot_bgcolor='#1a202c',  # Slightly lighter plot area
        font=dict(size=11),      # Slightly smaller font
        images=logo_images,
        xaxis=dict(showgrid=True, gridcolor='#2c3e50', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#2c3e50', zeroline=False),
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb")
        ),
        autosize=True
    )
    
    # 4. Add Quadrant Lines (League Averages)
    # These dashed lines indicate the average payroll and average wins.
    fig_quadrant.add_vline(x=avg_payroll, line_dash="dash", line_color="rgba(255,255,255,0.6)", line_width=1.5,
                          annotation_text="Avg Payroll", annotation_position="top right", annotation_font_color="#adb5bd")
    fig_quadrant.add_hline(y=avg_wins, line_dash="dash", line_color="rgba(255,255,255,0.6)", line_width=1.5,
                          annotation_text="Avg Wins", annotation_position="bottom right", annotation_font_color="#adb5bd")
    
    # Add Text Labels for Context (Quadrant Names)
    fig_quadrant.add_annotation(x=df_teams['Total_Payroll'].min(), y=df_teams['WINS'].max(), 
                               text="<b>ELITE</b>", showarrow=False, 
                               font=dict(color="#06d6a0", size=12, weight="bold"), xanchor="left", yanchor="top",
                               bgcolor="rgba(15, 22, 35, 0.7)", borderpad=4)
                               
    fig_quadrant.add_annotation(x=df_teams['Total_Payroll'].max(), y=df_teams['WINS'].max(), 
                               text="<b>CONTENDERS</b>", showarrow=False, 
                               font=dict(color="#ffd166", size=12, weight="bold"), xanchor="right", yanchor="top",
                               bgcolor="rgba(15, 22, 35, 0.7)", borderpad=4)
                               
    fig_quadrant.add_annotation(x=df_teams['Total_Payroll'].min(), y=df_teams['WINS'].min(), 
                               text="<b>REBUILDING</b>", showarrow=False, 
                               font=dict(color="#e4e6eb", size=10), xanchor="left", yanchor="bottom",
                               bgcolor="rgba(15, 22, 35, 0.7)", borderpad=4)
                               
    fig_quadrant.add_annotation(x=df_teams['Total_Payroll'].max(), y=df_teams['WINS'].min(), 
                               text="<b>DISASTER</b>", showarrow=False, 
                               font=dict(color="#ef476f", size=12, weight="bold"), xanchor="right", yanchor="bottom",
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
                    sizex=0.5, sizey=0.5,
                    xanchor="center", yanchor="middle",
                    layer="above"
                ))
    
    fig_grid.update_layout(
        # title='<b>Team Efficiency Rankings</b><br><sub style="color:#adb5bd">Sorted by Efficiency • Green = Good Value • Red = Overpaying</sub>',
        height=650,
        margin=dict(l=40, r=20, t=20, b=50),
        paper_bgcolor='#0f1623',
        plot_bgcolor='#1a202c',
        font=dict(size=11),
        autosize=True,
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
            showscale=False,  # Hide colorbar to match other chart width
            line=dict(width=1.5, color='rgba(255,255,255,0.4)'),
            opacity=0.95
        ),
        text=[f"<b>{n}</b><br>Gap: {g:.1f}" for n, g in 
              zip(filtered['player_name'], filtered['value_gap'] if 'value_gap' in filtered.columns else [0]*len(filtered))],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Calculate axis ranges for arrow positioning
    x_min = filtered['LEBRON'].min() if len(filtered) > 0 else -2
    x_max = filtered['LEBRON'].max() if len(filtered) > 0 else 8
    y_col = 'current_year_salary' if 'current_year_salary' in filtered.columns else 'total_contract_value'
    y_min = filtered[y_col].min() if len(filtered) > 0 else 0
    y_max = filtered[y_col].max() if len(filtered) > 0 else 50000000
    
    # Add axis indicator arrows with labels
    fig.update_layout(
        xaxis_title='<b>LEBRON Total</b>',
        yaxis_title='<b>Salary ($)</b>',
        height=500,
        template='plotly_dark',
        hovermode='closest',
        paper_bgcolor='#0f1623',
        plot_bgcolor='#1a202c',
        margin=dict(l=50, r=50, t=50, b=50),
        autosize=True,
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb")
        ),
        annotations=[
            # Right arrow - "Better Impact"
            dict(
                x=1.0, y=0.02,
                xref='paper', yref='paper',
                text='Better Impact  \u2192',
                showarrow=False,
                font=dict(size=11, color='#06d6a0', family='Arial'),
                xanchor='right'
            ),
            # Up arrow - "More Expensive"  
            dict(
                x=0.02, y=1.0,
                xref='paper', yref='paper',
                text='\u2191  More $',
                showarrow=False,
                font=dict(size=11, color='#ef476f', family='Arial'),
                xanchor='left', yanchor='top',
                textangle=0
            ),
        ]
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
            
            # Create images list
            # Commented out images as they don't position correctly on mobile
            # images = []
            # if 'PLAYER_ID' in top_under.columns:
            #     for i, row in enumerate(top_under.itertuples()):
            #         if pd.notna(row.PLAYER_ID):
            #             pid = int(row.PLAYER_ID)
            #             img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
            #             images.append(dict(
            #                 source=img_url,
            #                 xref="paper", yref="y",
            #                 x=-0.08, # Closer to chart edge for mobile
            #                 y=row.player_name,
            #                 sizex=0.12, sizey=0.7,
            #                 xanchor="right", yanchor="middle",
            #                 layer="above"
            #             ))

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
                margin=dict(l=150, r=20, t=20, b=50), # Reduced left margin
                autosize=True,
                showlegend=False,
                paper_bgcolor='#0f1623',
                plot_bgcolor='#1a202c',
                # images=images,  # Removed images for mobile compatibility
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
        margin=dict(l=50, r=20, t=20, b=50),
        autosize=True,
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


def create_age_ridge_plot(filtered):
    """
    Creates a Ridge Plot showing LEBRON distribution by age group.
    
    Ridge plots (also called joy plots) show distribution curves stacked vertically,
    making it easy to compare how player impact varies across age brackets.
    
    Args:
        filtered (pd.DataFrame): Filtered DataFrame of players.
        
    Returns:
        go.Figure: A Plotly ridge plot.
    """
    import plotly.figure_factory as ff
    
    fig = go.Figure()
    
    if len(filtered) == 0 or 'Age' not in filtered.columns or 'LEBRON' not in filtered.columns:
        fig.add_annotation(text="No data available", font=dict(size=14, color='#adb5bd'))
        fig.update_layout(height=500, template='plotly_dark', paper_bgcolor='#0f1623')
        return fig
    
    # Create age groups
    age_bins = [(19, 22, 'Rookies (19-22)'), (23, 25, 'Rising (23-25)'), 
                (26, 30, 'Prime (26-30)'), (31, 34, 'Veteran (31-34)'), (35, 45, 'Twilight (35+)')]
    
    colors = ['#06d6a0', '#2D96C7', '#ff6b35', '#ffd166', '#ef476f']
    
    # Build traces for each age group
    for i, (min_age, max_age, label) in enumerate(age_bins):
        group_data = filtered[(filtered['Age'] >= min_age) & (filtered['Age'] <= max_age)]['LEBRON']
        
        if len(group_data) >= 3:  # Need at least 3 points for KDE
            # Create KDE-like distribution using histogram
            hist_values, bin_edges = np.histogram(group_data, bins=20, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Offset each distribution vertically
            y_offset = i * 0.8
            
            # Add filled area
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=hist_values + y_offset,
                mode='lines',
                fill='tonexty' if i > 0 else 'tozeroy',
                fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.4)',
                line=dict(color=colors[i], width=2),
                name=f'{label} (n={len(group_data)})',
                hovertemplate=f'<b>{label}</b><br>LEBRON: %{{x:.2f}}<extra></extra>'
            ))
            
            # Add baseline for next trace
            if i < len(age_bins) - 1:
                fig.add_trace(go.Scatter(
                    x=bin_centers,
                    y=[y_offset + 0.8] * len(bin_centers),
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        xaxis_title='<b>LEBRON Impact</b>',
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        height=500,
        template='plotly_dark',
        paper_bgcolor='#0f1623',
        plot_bgcolor='#1a202c',
        margin=dict(l=50, r=20, t=30, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=11)
        ),
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb")
        )
    )
    
    # Add vertical line at 0 (league average)
    fig.add_vline(x=0, line_dash="dash", line_color="#6c757d", 
                  annotation_text="Avg", annotation_position="top")
    
    return fig


def create_player_beeswarm(filtered):
    """
    Creates a Beeswarm plot showing all players by LEBRON impact with colored dots.
    
    Args:
        filtered (pd.DataFrame): Filtered DataFrame of players.
        
    Returns:
        go.Figure: A Plotly beeswarm figure.
    """
    fig = go.Figure()
    
    if len(filtered) == 0 or 'LEBRON' not in filtered.columns:
        fig.add_annotation(text="No data available", font=dict(size=14, color='#adb5bd'))
        fig.update_layout(height=500, template='plotly_dark', paper_bgcolor='#0f1623')
        return fig
    
    df = filtered.copy()
    df = df.sort_values('LEBRON', ascending=True).reset_index(drop=True)
    
    # Create beeswarm-like y-positions to avoid overlap
    df['lebron_bin'] = pd.cut(df['LEBRON'], bins=30, labels=False)
    
    y_positions = []
    for bin_id in df['lebron_bin'].unique():
        bin_mask = df['lebron_bin'] == bin_id
        bin_count = bin_mask.sum()
        if bin_count > 1:
            offsets = np.linspace(-bin_count/2, bin_count/2, bin_count) * 0.35
        else:
            offsets = [0]
        
        for i, (idx, row) in enumerate(df[bin_mask].iterrows()):
            y_positions.append((idx, offsets[i % len(offsets)]))
    
    y_positions.sort(key=lambda x: x[0])
    df['y_offset'] = [yp[1] for yp in y_positions]
    
    # Color by value gap
    colors = []
    for vg in df['value_gap'] if 'value_gap' in df.columns else [0]*len(df):
        if vg > 10:
            colors.append('#06d6a0')  # Green - very underpaid
        elif vg > 0:
            colors.append('#2D96C7')  # Blue - slight value
        elif vg > -10:
            colors.append('#ffd166')  # Yellow - slight overpay
        else:
            colors.append('#ef476f')  # Red - overpaid
    
    value_gaps = df['value_gap'] if 'value_gap' in df.columns else [0]*len(df)
    salaries = df['current_year_salary'] if 'current_year_salary' in df.columns else [0]*len(df)
    
    fig.add_trace(go.Scatter(
        x=df['LEBRON'],
        y=df['y_offset'],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            opacity=0.85,
            line=dict(width=1, color='rgba(255,255,255,0.3)')
        ),
        text=[f"<b>{n}</b><br>LEBRON: {l:.2f}<br>Value Gap: {g:+.1f}<br>${s/1e6:.1f}M" 
              for n, l, g, s in zip(df['player_name'], df['LEBRON'], value_gaps, salaries)],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    # Add reference line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="#6c757d", line_width=1)
    
    fig.update_layout(
        xaxis=dict(
            title='<b>LEBRON Impact</b>',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        height=500,
        template='plotly_dark',
        paper_bgcolor='#0f1623',
        plot_bgcolor='#1a202c',
        margin=dict(l=50, r=50, t=50, b=50),
        autosize=True,
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb")
        )
    )
    
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
            
            # Create images list
            # Commented out images as they don't position correctly on mobile
            # images = []
            # if 'PLAYER_ID' in top_over.columns:
            #     for i, row in enumerate(top_over.itertuples()):
            #         if pd.notna(row.PLAYER_ID):
            #             pid = int(row.PLAYER_ID)
            #             img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
            #             images.append(dict(
            #                 source=img_url,
            #                 xref="paper", yref="y",
            #                 x=-0.08, # Closer to chart edge for mobile
            #                 y=row.player_name,
            #                 sizex=0.12, sizey=0.7,
            #                 xanchor="right", yanchor="middle",
            #                 layer="above"
            #             ))

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
                margin=dict(l=150, r=20, t=20, b=50), # Reduced left margin
                autosize=True,
                showlegend=False,
                paper_bgcolor='#0f1623',
                plot_bgcolor='#1a202c',
                # images=images,  # Removed images for mobile compatibility
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
        top_data = data.nlargest(20, 'value_gap')[['player_name', 'current_year_salary', 'LEBRON', 'value_gap', 'PLAYER_ID']].copy()
        text_color = "#06d6a0"
    elif table_type == 'overpaid':
        data = filtered[filtered['value_gap'] < 0]
        if len(data) == 0: return html.P("No overpaid players in filter")
        top_data = data.nsmallest(20, 'value_gap')[['player_name', 'current_year_salary', 'LEBRON', 'value_gap', 'PLAYER_ID']].copy()
        text_color = "#ef476f"
    else:
        return html.P("Invalid table type")

    # Clean up player names and add Face Markdown
    top_data['player_name'] = top_data['player_name'].apply(
        lambda x: str(x).strip() if pd.notna(x) else 'Unknown'
    )
    
    def get_face_markdown(pid):
        if pd.notna(pid):
            url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(pid)}.png"
            # Use HTML for better size control
            return f'<img src="{url}" style="height: 30px; width: auto; border-radius: 50%;" />'
        return ""
        
    top_data['Face'] = top_data['PLAYER_ID'].apply(get_face_markdown)
    
    return dash_table.DataTable(
        data=top_data.to_dict('records'),
        columns=[
            {'name': '', 'id': 'Face', 'presentation': 'markdown'},
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
            'borderBottom': '1px solid #2c3e50',
            'verticalAlign': 'middle' # Align text with images
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
            },
            # Face column width
            {
                'if': {'column_id': 'Face'},
                'width': '40px',
                'padding': '0px'
            }
        ],
        page_action='none',
        markdown_options={'html': True} # Enable HTML in markdown
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
        ['player_name', 'archetype', 'current_year_salary', 'LEBRON', 'value_gap', 'PLAYER_ID']
    ].copy()
    
    # Clean up player names and archetypes
    all_players_data['player_name'] = all_players_data['player_name'].apply(
        lambda x: str(x).strip() if pd.notna(x) else 'Unknown'
    )
    all_players_data['archetype'] = all_players_data['archetype'].apply(
        lambda x: str(x).strip() if pd.notna(x) else 'Unknown'
    )
    
    # Add Face HTML
    def get_face_html(pid):
        if pd.notna(pid):
            url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(pid)}.png"
            return f'<img src="{url}" style="height: 30px; width: auto; border-radius: 50%;" />'
        return ""
        
    all_players_data['Face'] = all_players_data['PLAYER_ID'].apply(get_face_html)
    
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
        },
        # Face column width
        {
            'if': {'column_id': 'Face'},
            'width': '40px',
            'padding': '0px'
        }
    ]
    
    return dash_table.DataTable(
        data=all_players_data.to_dict('records'),
        columns=[
            {'name': '', 'id': 'Face', 'presentation': 'markdown'},
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
            'borderBottom': '1px solid #2c3e50',
            'verticalAlign': 'middle' # Align text with images
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
        page_action='none',
        markdown_options={'html': True}
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
    
    # Get team colors
    color1 = NBA_TEAM_COLORS.get(team1_name, '#ff6b35')
    color2 = NBA_TEAM_COLORS.get(team2_name, '#2D96C7')
    
    # Convert hex to rgba for fill
    def hex_to_rgba(hex_color, alpha=0.25):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f'rgba({r}, {g}, {b}, {alpha})'
        return f'rgba(100, 100, 100, {alpha})'

    fill1 = hex_to_rgba(color1)
    fill2 = hex_to_rgba(color2)
    
    fig = go.Figure()
    
    # Add Team 1
    fig.add_trace(go.Scatterpolar(
        r=values_team1,
        theta=categories,
        fill='toself',
        name=team1_name,
        line=dict(color=color1, width=3),
        fillcolor=fill1,
        hovertemplate='<b>%{theta}</b><br>' + team1_name + ': %{r:.1f}th percentile<extra></extra>',
        marker=dict(size=8, color=color1)
    ))
    
    # Add Team 2
    fig.add_trace(go.Scatterpolar(
        r=values_team2,
        theta=categories,
        fill='toself',
        name=team2_name,
        line=dict(color=color2, width=3),
        fillcolor=fill2,
        hovertemplate='<b>%{theta}</b><br>' + team2_name + ': %{r:.1f}th percentile<extra></extra>',
        marker=dict(size=8, color=color2)
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


def create_player_radar_mini(stats, position='Wing', player_name='Player'):
    """
    Creates a compact hexagonal radar chart for a player card showing their archetype characteristics.
    
    Args:
        stats (dict): Dictionary of player statistics
        position (str): Player position group ('Guard', 'Wing', 'Big')
        player_name (str): Player's name for tooltip
        
    Returns:
        go.Figure: Compact Plotly radar chart
    """
    # Define the 6 categories for hexagonal radar
    categories = [
        'Scoring',      # PTS
        'Playmaking',   # AST%, TOV_AST_RATIO
        'Shooting',     # rTS, 3PA_RATE, FT%
        'Rebounding',   # REB, DREB%, OREB%
        'Defense',      # STL, BLK, DEF_RATING
        'Efficiency'    # USG%, FG2%
    ]
    
    # Calculate percentile values (0-100) for each category
    # These are rough estimates - in production, you'd calculate actual percentiles
    values = []
    
    # 1. Scoring (based on PTS per 100)
    pts = stats.get('PTS', 0)
    scoring_score = min(100, (pts / 35) * 100)  # 35 PTS/100 = 100th percentile
    values.append(scoring_score)
    
    # 2. Playmaking (AST% primarily)
    ast_pct = stats.get('AST_PCT', 0)
    playmaking_score = min(100, (ast_pct / 40) * 100)  # 40 AST% = 100th percentile
    values.append(playmaking_score)
    
    # 3. Shooting (composite of rTS and 3PA_RATE)
    rts = stats.get('rTS', 0)
    shooting_score = max(0, min(100, 50 + rts))  # rTS of +5 = 100th percentile
    values.append(shooting_score)
    
    # 4. Rebounding (REB per 100)
    reb = stats.get('REB', 0)
    rebounding_score = min(100, (reb / 15) * 100)  # 15 REB/100 = 100th percentile
    values.append(rebounding_score)
    
    # 5. Defense (composite of STL, BLK)
    stl = stats.get('STL', 0)
    blk = stats.get('BLK', 0)
    defense_score = min(100, ((stl + blk) / 4) * 100)  # 4 combined = 100th percentile
    values.append(defense_score)
    
    # 6. Efficiency (USG% and rTS combined)
    usg = stats.get('USG_PCT', 0)
    efficiency_score = min(100, (usg / 35) * 100)  # 35 USG% = 100th percentile
    values.append(efficiency_score)
    
    # Close the radar loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    # Determine color - use custom color if provided (for match quality), otherwise position-based
    if position and position.startswith('#'):
        # It's actually a color code passed as position parameter
        line_color = position
        # Convert hex to rgba for fill
        hex_color = position.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        fill_color = f'rgba({r}, {g}, {b}, 0.3)'
    elif position.lower() == 'guard':
        line_color = '#2D96C7'  # Blue
        fill_color = 'rgba(45, 150, 199, 0.3)'
    elif position.lower() == 'big':
        line_color = '#ef476f'  # Red
        fill_color = 'rgba(239, 71, 111, 0.3)'
    else:  # Wing or default
        line_color = '#06d6a0'  # Green
        fill_color = 'rgba(6, 214, 160, 0.3)'
    
    fig = go.Figure()
    
    # Add player trace
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        line=dict(color=line_color, width=2),
        fillcolor=fill_color,
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.0f}/100<extra></extra>',
        marker=dict(size=4, color=line_color),
        showlegend=False
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(255, 255, 255, 0.1)',
                gridwidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color='#e4e6eb', family='Inter, sans-serif'),
                gridcolor='rgba(255, 255, 255, 0.15)',
                linecolor='rgba(255, 255, 255, 0.15)',
                gridwidth=1
            ),
            bgcolor='rgba(0, 0, 0, 0.2)'
        ),
        height=220,  # Increased to prevent label cutoff
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor=line_color,
            font=dict(color="#e4e6eb", size=10)
        )
    )
    
    return fig


# ============================================================================
# LINEUP CHEMISTRY VISUALIZATIONS
# ============================================================================

def create_lineup_bar_chart(df, title='Top Lineups', color='#06d6a0', metric='PLUS_MINUS'):
    """
    Creates a horizontal bar chart showing lineup performance.
    
    Args:
        df (pd.DataFrame): Lineup data with GROUP_NAME and metric columns.
        title (str): Chart title.
        color (str): Bar color.
        metric (str): Metric to display ('PLUS_MINUS', 'W_PCT', 'PTS', etc.)
        
    Returns:
        go.Figure: Plotly bar chart.
    """
    if df.empty or 'GROUP_NAME' not in df.columns:
        fig = go.Figure().add_annotation(text="No lineup data available")
        fig.update_layout(height=400, template='plotly_dark', paper_bgcolor='#0f1623')
        return fig
    
    # Use PLUS_MINUS if the requested metric isn't available
    if metric not in df.columns:
        metric = 'PLUS_MINUS' if 'PLUS_MINUS' in df.columns else df.columns[0]
    
    # Take top 10 and reverse for horizontal bar chart
    display_df = df.head(10).iloc[::-1].copy()
    
    # Format player names (shorten if needed)
    def format_lineup_name(name):
        """Formats lineup names to be more readable."""
        if not isinstance(name, str):
            return str(name)
        # Split by ' - ' and take last names
        players = name.split(' - ')
        formatted = []
        for player in players:
            parts = player.strip().split(' ')
            if len(parts) >= 2:
                # First initial + last name
                formatted.append(f"{parts[0][0]}. {parts[-1]}")
            else:
                formatted.append(player.strip())
        return ' + '.join(formatted)
    
    display_df['Display_Name'] = display_df['GROUP_NAME'].apply(format_lineup_name)
    
    # Determine color based on values (green for positive, red for negative)
    colors = [color if v >= 0 else '#ef476f' for v in display_df[metric]]
    
    fig = go.Figure(go.Bar(
        x=display_df[metric],
        y=display_df['Display_Name'],
        orientation='h',
        marker=dict(
            color=colors,
            opacity=0.9,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f"{v:+.1f}" for v in display_df[metric]],
        textposition='outside',
        textfont=dict(size=11, color='#e4e6eb'),
        customdata=np.stack([
            display_df['MIN'].values if 'MIN' in display_df.columns else [0]*len(display_df),
            display_df['GP'].values if 'GP' in display_df.columns else [0]*len(display_df),
            display_df['TEAM_ABBREVIATION'].values if 'TEAM_ABBREVIATION' in display_df.columns else ['']*len(display_df)
        ], axis=-1),
        hovertemplate='<b>%{y}</b><br>Net Rating: %{x:+.1f}<br>Minutes: %{customdata[0]:.0f}<br>Games: %{customdata[1]}<br>Team: %{customdata[2]}<extra></extra>'
    ))
    
    fig.update_layout(
        height=450,
        template='plotly_dark',
        margin=dict(l=200, r=60, t=10, b=40),
        paper_bgcolor='#0f1623',
        plot_bgcolor='#1a202c',
        showlegend=False,
        xaxis=dict(
            title=f'<b>{metric.replace("_", " ")}</b>',
            gridcolor='#2c3e50',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=2
        ),
        yaxis=dict(
            tickfont=dict(size=11),
            gridcolor='#2c3e50'
        ),
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb")
        )
    )
    
    return fig


def create_lineup_scatter(df_best, df_worst, title='Lineup Chemistry Map'):
    """
    Creates a scatter plot showing Plus/Minus vs Minutes for lineups.
    
    Args:
        df_best (pd.DataFrame): Best performing lineups.
        df_worst (pd.DataFrame): Worst performing lineups.
        title (str): Chart title.
        
    Returns:
        go.Figure: Plotly scatter plot.
    """
    fig = go.Figure()
    
    # Helper to format names
    def format_lineup_name(name):
        if not isinstance(name, str):
            return str(name)
        players = name.split(' - ')
        formatted = []
        for player in players:
            parts = player.strip().split(' ')
            if len(parts) >= 2:
                formatted.append(f"{parts[0][0]}. {parts[-1]}")
            else:
                formatted.append(player.strip())
        return ' + '.join(formatted)
    
    # Add best lineups (green)
    if not df_best.empty and 'PLUS_MINUS' in df_best.columns:
        df_best = df_best.copy()
        df_best['Display_Name'] = df_best['GROUP_NAME'].apply(format_lineup_name)
        
        fig.add_trace(go.Scatter(
            x=df_best['MIN'] if 'MIN' in df_best.columns else range(len(df_best)),
            y=df_best['PLUS_MINUS'],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#06d6a0',
                opacity=0.8,
                line=dict(width=2, color='rgba(255,255,255,0.5)')
            ),
            text=df_best['Display_Name'],
            textposition='top center',
            textfont=dict(size=8, color='#06d6a0'),
            name='Best Lineups',
            hovertemplate='<b>%{text}</b><br>+/-: %{y:+.1f}<br>Minutes: %{x:.0f}<extra></extra>'
        ))
    
    # Add worst lineups (red)
    if not df_worst.empty and 'PLUS_MINUS' in df_worst.columns:
        df_worst = df_worst.copy()
        df_worst['Display_Name'] = df_worst['GROUP_NAME'].apply(format_lineup_name)
        
        fig.add_trace(go.Scatter(
            x=df_worst['MIN'] if 'MIN' in df_worst.columns else range(len(df_worst)),
            y=df_worst['PLUS_MINUS'],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#ef476f',
                opacity=0.8,
                line=dict(width=2, color='rgba(255,255,255,0.5)')
            ),
            text=df_worst['Display_Name'],
            textposition='bottom center',
            textfont=dict(size=8, color='#ef476f'),
            name='Worst Lineups',
            hovertemplate='<b>%{text}</b><br>+/-: %{y:+.1f}<br>Minutes: %{x:.0f}<extra></extra>'
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1)
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        margin=dict(l=60, r=20, t=20, b=60),
        paper_bgcolor='#0f1623',
        plot_bgcolor='#1a202c',
        xaxis=dict(
            title='<b>Total Minutes Together</b>',
            gridcolor='#2c3e50'
        ),
        yaxis=dict(
            title='<b>Plus/Minus</b> (higher = better)',
            gridcolor='#2c3e50'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(26, 35, 50, 0.8)',
            bordercolor='#3a4555',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="#1a2332",
            bordercolor="#ff6b35",
            font=dict(color="#e4e6eb")
        )
    )
    
    return fig


def create_lineup_table(df, table_type='best'):
    """
    Creates a styled data table for lineup data.
    
    Args:
        df (pd.DataFrame): Lineup data.
        table_type (str): 'best' or 'worst' to determine styling.
        
    Returns:
        dash_table.DataTable: Styled table component.
    """
    if df.empty:
        return html.P("No lineup data available", className="text-muted text-center")
    
    # Format lineup names
    def format_lineup_name(name):
        if not isinstance(name, str):
            return str(name)
        players = name.split(' - ')
        formatted = []
        for player in players:
            parts = player.strip().split(' ')
            if len(parts) >= 2:
                formatted.append(f"{parts[0][0]}. {parts[-1]}")
            else:
                formatted.append(player.strip())
        return ' + '.join(formatted)
    
    display_df = df.head(10).copy()
    display_df['Lineup'] = display_df['GROUP_NAME'].apply(format_lineup_name)
    
    text_color = '#06d6a0' if table_type == 'best' else '#ef476f'
    
    # Build columns dynamically based on what's available
    columns = [{'name': 'Lineup', 'id': 'Lineup', 'type': 'text'}]
    
    if 'TEAM_ABBREVIATION' in display_df.columns:
        columns.append({'name': 'Team', 'id': 'TEAM_ABBREVIATION', 'type': 'text'})
    if 'MIN' in display_df.columns:
        columns.append({'name': 'MIN', 'id': 'MIN', 'type': 'numeric', 'format': {'specifier': '.0f'}})
    if 'GP' in display_df.columns:
        columns.append({'name': 'GP', 'id': 'GP', 'type': 'numeric'})
    if 'PLUS_MINUS' in display_df.columns:
        columns.append({'name': '+/-', 'id': 'PLUS_MINUS', 'type': 'numeric', 'format': {'specifier': '+.1f'}})
    if 'W_PCT' in display_df.columns:
        columns.append({'name': 'W%', 'id': 'W_PCT', 'type': 'numeric', 'format': {'specifier': '.3f'}})
    if 'PTS' in display_df.columns:
        columns.append({'name': 'PTS', 'id': 'PTS', 'type': 'numeric', 'format': {'specifier': '.1f'}})
    
    return dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=columns,
        style_table={'overflowX': 'auto'},
        style_cell={
            'backgroundColor': '#1a2332',
            'color': '#e4e6eb',
            'textAlign': 'left',
            'padding': '8px 12px',
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
            'fontSize': '12px',
            'padding': '8px 12px'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'PLUS_MINUS'},
                'color': text_color,
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Lineup'},
                'fontWeight': '600',
                'maxWidth': '200px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            }
        ],
        page_action='none',
        sort_action='native'
    )


# =============================================================================
# DIAMOND FINDER VISUALIZATIONS
# =============================================================================

def create_mini_radar(target_lebron, target_o, target_d, match_lebron, match_o, match_d):
    """
    Creates a mini radar chart comparing target and match player LEBRON metrics.
    
    Args:
        target_lebron, target_o, target_d: Target player's LEBRON, O-LEBRON, D-LEBRON
        match_lebron, match_o, match_d: Match player's LEBRON, O-LEBRON, D-LEBRON
        
    Returns:
        go.Figure: A compact radar chart
    """
    categories = ['Overall', 'Offense', 'Defense']
    
    # Normalize values to 0-10 scale for visual comparison
    # LEBRON typically ranges from -4 to +8
    def normalize(val):
        return max(0, min(10, (val + 4) * 0.8))
    
    target_vals = [normalize(target_lebron), normalize(target_o), normalize(target_d)]
    match_vals = [normalize(match_lebron), normalize(match_o), normalize(match_d)]
    
    fig = go.Figure()
    
    # Target player (outline only)
    fig.add_trace(go.Scatterpolar(
        r=target_vals + [target_vals[0]],  # Close the shape
        theta=categories + [categories[0]],
        fill='none',
        line=dict(color='#ff6b35', width=2, dash='dot'),
        name='Target',
        hoverinfo='skip'
    ))
    
    # Match player (filled)
    fig.add_trace(go.Scatterpolar(
        r=match_vals + [match_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(6, 214, 160, 0.3)',
        line=dict(color='#06d6a0', width=2),
        name='Match',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[0, 10]
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color='#adb5bd'),
                rotation=90,
                direction='clockwise'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=120,
        width=120
    )
    
    return fig


def _create_stat_comparison(label, match_val, target_val):
    """Creates a small stat comparison row for the replacement card."""
    from dash import html
    
    if match_val == 0 and target_val == 0:
        return html.Div()
    
    # Format as percentage
    match_str = f"{match_val:.1%}" if match_val > 0 else "N/A"
    target_str = f"{target_val:.1%}" if target_val > 0 else "N/A"
    
    # Determine color based on closeness
    if match_val > 0 and target_val > 0:
        diff = abs(match_val - target_val)
        if diff < 0.03:
            color = "#06d6a0"  # Very close
        elif diff < 0.06:
            color = "#2D96C7"  # Close
        elif diff < 0.10:
            color = "#ffd166"  # Moderate
        else:
            color = "#adb5bd"  # Different
    else:
        color = "#6c757d"
    
    return html.Div([
        html.Span(f"{label}: ", style={"color": "#6c757d", "fontSize": "9px"}),
        html.Span(match_str, style={"color": color, "fontWeight": "600", "fontSize": "10px"}),
        html.Span(f" ({target_str})", style={"color": "#4a5568", "fontSize": "9px"})
    ], style={"marginBottom": "2px"})


def create_replacement_card(replacement, target_name, target_salary):
    """
    Creates a styled card for a replacement player in the Diamond Finder.
    
    Shows player headshot, match score, mini radar comparison, archetype, and savings.
    
    Args:
        replacement (dict): Replacement player data from find_replacement_players()
        target_name (str): Name of the target player being replaced
        target_salary (float): Target player's salary
        
    Returns:
        dbc.Card: A styled Bootstrap card component
    """
    from dash import html, dcc
    import dash_bootstrap_components as dbc
    
    player_id = replacement.get('PLAYER_ID')
    img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png" if pd.notna(player_id) else ""
    
    # Determine match quality color
    score = replacement['match_score']
    if score >= 90:
        score_color = '#06d6a0'  # Excellent
        score_label = 'Excellent Match'
    elif score >= 75:
        score_color = '#2D96C7'  # Good
        score_label = 'Good Match'
    elif score >= 60:
        score_color = '#ffd166'  # Decent
        score_label = 'Decent Match'
    else:
        score_color = '#adb5bd'  # Fair
        score_label = 'Fair Match'
    
    # Create mini radar
    radar_fig = create_mini_radar(
        replacement['target_LEBRON'], replacement['target_O_LEBRON'], replacement['target_D_LEBRON'],
        replacement['LEBRON'], replacement['O-LEBRON'], replacement['D-LEBRON']
    )
    
    card = dbc.Card([
        dbc.CardBody([
            # Top row: Headshot + Match Score
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src=img_url,
                        style={
                            "width": "60px", "height": "60px", 
                            "borderRadius": "50%", 
                            "border": f"3px solid {score_color}",
                            "objectFit": "cover"
                        }
                    ) if img_url else html.Div(style={"width": "60px", "height": "60px"})
                ], width=4, className="text-center"),
                dbc.Col([
                    html.Div(replacement['player_name'], style={
                        "fontWeight": "700", "fontSize": "14px", "color": "#e4e6eb",
                        "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"
                    }),
                    html.Div([
                        html.Span(f"{score:.0f}%", style={
                            "color": score_color, "fontWeight": "700", "fontSize": "18px"
                        }),
                        html.Span(f" {score_label}", style={
                            "color": "#6c757d", "fontSize": "10px", "marginLeft": "4px"
                        })
                    ]),
                    html.Div(f"${replacement['salary']/1e6:.1f}M", style={
                        "color": "#06d6a0", "fontWeight": "600", "fontSize": "13px"
                    })
                ], width=8)
            ], className="mb-2"),
            
            # Middle: Advanced stats comparison (style similarity)
            dbc.Row([
                dbc.Col([
                    # Advanced stats comparison
                    _create_stat_comparison("USG", replacement.get('USG_PCT', 0), replacement.get('target_USG', 0)),
                    _create_stat_comparison("AST%", replacement.get('AST_PCT', 0), replacement.get('target_AST', 0)),
                    _create_stat_comparison("TS%", replacement.get('TS_PCT', 0), replacement.get('target_TS', 0)),
                ], width=6),
                dbc.Col([
                    # Archetype badges
                    html.Div([
                        html.Span(replacement['archetype'], style={
                            "backgroundColor": "rgba(45, 150, 199, 0.3)",
                            "color": "#2D96C7",
                            "padding": "2px 6px",
                            "borderRadius": "4px",
                            "fontSize": "9px",
                            "display": "block",
                            "marginBottom": "4px"
                        })
                    ]),
                    html.Div(replacement['defense_role'], style={
                        "fontSize": "9px", "color": "#6c757d", "marginBottom": "8px"
                    }),
                    # LEBRON comparison (smaller)
                    html.Div([
                        html.Span(f"LEBRON: {replacement['LEBRON']:.1f}", style={
                            "color": "#06d6a0" if replacement['LEBRON'] > 0 else "#ffd166", 
                            "fontSize": "10px", "fontWeight": "600"
                        }),
                        html.Span(f" vs {replacement['target_LEBRON']:.1f}", style={
                            "color": "#ff6b35", "fontSize": "10px"
                        })
                    ])
                ], width=6, className="d-flex flex-column justify-content-center")
            ], className="mb-2"),
            
            # Bottom: Savings highlight
            html.Div([
                html.Span(f"Save ${replacement['savings']/1e6:.1f}M ", style={
                    "color": "#06d6a0", "fontWeight": "700", "fontSize": "12px"
                }),
                html.Span(f"({replacement['savings_pct']:.0f}% cheaper)", style={
                    "color": "#6c757d", "fontSize": "10px"
                })
            ], className="text-center mt-2", style={
                "backgroundColor": "rgba(6, 214, 160, 0.1)",
                "padding": "6px",
                "borderRadius": "4px"
            })
        ], style={"padding": "12px"})
    ], style={
        "backgroundColor": "#151b26",
        "border": f"1px solid {score_color}",
        "borderRadius": "8px",
        "minWidth": "220px",
        "maxWidth": "260px",
        "flexShrink": "0"
    })
    
    return card


def create_diamond_finder_results(replacements, target_name, target_salary, target_lebron):
    """
    Creates the full Diamond Finder results display.
    
    Args:
        replacements (list): List of replacement dicts from find_replacement_players()
        target_name (str): Target player name
        target_salary (float): Target player salary
        target_lebron (float): Target player LEBRON score
        
    Returns:
        html.Div: Container with all replacement cards
    """
    from dash import html
    
    if not replacements:
        return html.Div([
            html.P("No cheaper replacements found for this player.", 
                   className="text-muted text-center py-4"),
            html.P("Try selecting a higher-paid player.", 
                   className="text-muted text-center", style={"fontSize": "12px"})
        ])
    
    cards = [create_replacement_card(r, target_name, target_salary) for r in replacements]
    
    return html.Div([
        # Header
        html.Div([
            html.Span(f"Found {len(replacements)} replacements for ", style={"color": "#adb5bd"}),
            html.Span(target_name, style={"color": "#ff6b35", "fontWeight": "600"}),
            html.Span(f" (${target_salary/1e6:.1f}M, LEBRON: {target_lebron:.2f})", 
                     style={"color": "#6c757d", "fontSize": "12px"})
        ], className="mb-3"),
        
        # Cards row (horizontal scrollable)
        html.Div(cards, style={
            "display": "flex",
            "overflowX": "auto",
            "paddingBottom": "10px",
            "gap": "12px"
        })
    ])
