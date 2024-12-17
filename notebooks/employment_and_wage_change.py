"""
employment_and_wage_change: Top 5 Employment and Wage Changes Across Industries

This module contains functions to display top 10 employee count and wage change across industries.
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

from data_preprocessing import process_wage_and_employment_data

common_df = process_wage_and_employment_data()

common_df['annual_2019'] = pd.to_numeric(common_df['annual_2019'], errors='coerce')
common_df['annual_2023'] = pd.to_numeric(common_df['annual_2023'], errors='coerce')

common_df = common_df.dropna(subset=['annual_2019', 'annual_2023'])

common_df['wage_percent_change'] = (
    abs(common_df['annual_2023'] - common_df['annual_2019']) / common_df['annual_2019'] * 100
)

common_df['percent_change'] = (
    abs(common_df['annual_2023'] - common_df['annual_2019']) / common_df['annual_2019'] * 100
)

employment_and_wage_change = dash.Dash(__name__)

employment_and_wage_change.layout = html.Div([
    html.H1("Employment and Wage Change Dashboard"),
    dcc.Tabs([
        dcc.Tab(label='Total Employment Change', children=[
            html.Div([
                dcc.Dropdown(
                    id='department-dropdown-emp',
                    options=[{'label': dept, 'value': dept}
                             for dept in common_df['department_name'].unique()],
                    value=common_df['department_name'].unique()[0],
                    clearable=False,
                    style={'width': '50%', 'margin': 'auto'}
                ),
                html.Div(id='top5-jobs-container-emp', style={
                    'display': 'flex',
                    'justify-content': 'space-between',
                    'width': '90%',
                    'margin': 'auto'
                })
            ])
        ]),
        dcc.Tab(label='Wage Change', children=[
            html.Div([
                dcc.Dropdown(
                    id='department-dropdown-wage',
                    options=[{'label': dept, 'value': dept}
                             for dept in common_df['department_name'].unique()],
                    value=common_df['department_name'].unique()[0],
                    clearable=False,
                    style={'width': '50%', 'margin': 'auto'}
                ),
                html.Div(id='top5-jobs-container-wage', style={
                    'display': 'flex',
                    'justify-content': 'space-between',
                    'width': '90%',
                    'margin': 'auto'
                })
            ])
        ])
    ])
])

@employment_and_wage_change.callback(
    Output('top5-jobs-container-emp', 'children'),
    [Input('department-dropdown-emp', 'value')]
)
def update_employment_dashboard(selected_department):
    """
    Update the employment change dashboard for the selected department.

    This function filters the data based on the selected department 
    and calculates the top 5 job titles with the highest and lowest 
    employment percentage changes.

    Args:
        selected_department (str): The department name selected from the dropdown.

    Returns:
        list: A list of two `dcc.Graph` components displaying the top 5 
        jobs with the highest and lowest employment changes as bar charts.
    """
    filtered_df = common_df[common_df['department_name'] == selected_department]

    filtered_df_unique = filtered_df.drop_duplicates(subset='occ_title')

    top5_increase = filtered_df_unique.nlargest(5, 'percent_change')
    top5_decrease = filtered_df_unique.nsmallest(5, 'percent_change')

    fig_increase = px.bar(
        top5_increase,
        x='occ_title',
        y='percent_change',
        title='Top 5 Jobs with Highest Employment Increase',
        labels={'percent_change': '% Increase'},
        color_discrete_sequence=['green'],
        text='percent_change'  # Adding text to the bars to show the actual value
    )
    fig_increase.update_layout(height=500, width=600, xaxis_tickangle=-45,
                                margin={"t": 50, "l": 50, "r": 50, "b": 150})
    fig_increase.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    fig_decrease = px.bar(
        top5_decrease,
        x='occ_title',
        y='percent_change',
        title='Top 5 Jobs with Highest Employment Decrease',
        labels={'percent_change': '% Decrease'},
        color_discrete_sequence=['red'],
        text='percent_change'  # Adding text to the bars to show the actual value
    )
    fig_decrease.update_layout(height=500, width=600, xaxis_tickangle=-45,
                                margin={"t": 50, "l": 50, "r": 50, "b": 150})
    fig_decrease.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    return [
        dcc.Graph(figure=fig_increase, style={'flex': '1 1 48%', 'margin': '10px'}),
        dcc.Graph(figure=fig_decrease, style={'flex': '1 1 48%', 'margin': '10px'})
    ]

@employment_and_wage_change.callback(
    Output('top5-jobs-container-wage', 'children'),
    [Input('department-dropdown-wage', 'value')]
)
def update_wage_dashboard(selected_department):
    """
    Update the wage change dashboard for the selected department.

    This function filters the data based on the selected department 
    and calculates the top 5 job titles with the highest and lowest 
    wage percentage changes.

    Args:
        selected_department (str): The department name selected from the dropdown.

    Returns:
        list: A list of two `dcc.Graph` components displaying the top 5 jobs
        with the highest and lowest wage changes as bar charts.
    """
    filtered_df = common_df[common_df['department_name'] == selected_department]

    filtered_df_unique = filtered_df.drop_duplicates(subset='occ_title')

    top5_increase = filtered_df_unique.nlargest(5, 'wage_percent_change')
    top5_decrease = filtered_df_unique.nsmallest(5, 'wage_percent_change')

    fig_increase = px.bar(
        top5_increase,
        x='occ_title',
        y='wage_percent_change',
        title='Top 5 Jobs with Highest Wage Increase',
        labels={'wage_percent_change': '% Wage Increase'},
        color_discrete_sequence=['green'],
        text='wage_percent_change'  # Adding text to the bars to show the actual value
    )
    fig_increase.update_layout(height=400, width=600, xaxis_tickangle=-45,
                                margin={"t": 50, "l": 50, "r": 50, "b": 150})
    fig_increase.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    fig_decrease = px.bar(
        top5_decrease,
        x='occ_title',
        y='wage_percent_change',
        title='Top 5 Jobs with Highest Wage Decrease',
        labels={'wage_percent_change': '% Wage Decrease'},
        color_discrete_sequence=['red'],
        text='wage_percent_change'  # Adding text to the bars to show the actual value
    )
    fig_decrease.update_layout(height=400, width=600, xaxis_tickangle=-45,
                                margin={"t": 50, "l": 50, "r": 50, "b": 150})
    fig_decrease.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    return [
        dcc.Graph(figure=fig_increase, style={'flex': '1 1 50%', 'margin': '10px'}),
        dcc.Graph(figure=fig_decrease, style={'flex': '1 1 50%', 'margin': '10px'})
    ]

if __name__ == '__main__':
    employment_and_wage_change.run_server(debug=True, port=8059)
