"""
work_trends_2029: Top 10 Jobs with the Highest Employment Change: 
Comparing 2023 and Projected 2029 Values

This module contains functions for an app to show bargraph of top jobs 
with highest projected job growth and decline between 2023 and 2029.
"""
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from data_preprocessing import load_and_clean_skills_occupation_data

# from visualisation import get_top_5_predictions

work_trends_2029 = dash.Dash(__name__)

_, dropdown_df = load_and_clean_skills_occupation_data()

dropdown_df.rename(columns={
    '2023 National Employment Matrix code': 'occ_code',
    '2023 National Employment Matrix title': 'occ_title'
}, inplace=True)

dropdown_df = dropdown_df.iloc[1:, :]

work_trends_2029.layout = html.Div([
    html.H1("Employment Change 2023 - 2029 Dashboard"),

    html.Div([
        html.Label("Select Job Prefixes"),
        dcc.Dropdown(
            id='prefix-dropdown',
            multi=True,
            style={'width': '80%', 'margin': 'auto'}
        ),
    ], style={'width': '80%', 'margin': 'auto', 'padding': '20px'}),

    html.Div([
        dcc.Graph(id='employment-change-graph-increase',
                  style={'flex': '1 1 48%', 'margin': '10px'}),
        dcc.Graph(id='employment-change-graph-decrease',
                  style={'flex': '1 1 48%', 'margin': '10px'})
    ], style={'display': 'flex', 'justify-content': 'space-between',
              'width': '90%', 'margin': 'auto'})
])

@work_trends_2029.callback(
    [
        Output('employment-change-graph-increase', 'figure'),
        Output('employment-change-graph-decrease', 'figure'),
        Output('prefix-dropdown', 'options')
    ],
    [Input('prefix-dropdown', 'value')]
)
def update_dashboard(selected_prefixes):
    """
    Updates the dashboard with employment change graphs and dropdown 
    options based on selected job prefixes.

    Args:
        selected_prefixes (list or None): A list of selected job code 
        prefixes. If None, all prefixes are used.

    Returns:
        tuple: A tuple containing:
            - A Plotly figure for the employment increase graph.
            - A Plotly figure for the employment decrease graph.
            - A list of dropdown options based on selected job prefixes.
    """

    # Get the top 5 predicted job titles dynamically
    # top_5_predictions = get_top_5_predictions()

    result_widget_options = [
        {'label': f"{code} - {title}", 'value': code}
        for code, title in zip(dropdown_df['occ_code'].str[:2].unique(),
                               dropdown_df.groupby('occ_code')['occ_title'].first())
        # if code in top_5_predictions
    ]

    if not selected_prefixes:
        selected_prefixes = dropdown_df['occ_code'].str[:2].unique()

    common_occupation_df, _ = load_and_clean_skills_occupation_data()
    filtered_df = common_occupation_df[common_occupation_df['occ_code']\
                                       .str[:2].isin(selected_prefixes)]

    top10_increase = filtered_df.nlargest(10, 'percent_change')
    top10_decrease = filtered_df.nsmallest(10, 'percent_change')

    fig_increase = px.bar(
        top10_increase,
        x='occ_title',
        y='percent_change',
        title='Top 10 Jobs with Highest Employment Increase',
        labels={'percent_change': '% Increase'},
        color_discrete_sequence=['green'],
        text='percent_change'
    )
    fig_increase.update_layout(height=600, width=800,
                               xaxis_tickangle=-45, margin={"t": 50, "l": 50, "r": 50, "b": 150})
    fig_increase.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    fig_decrease = px.bar(
        top10_decrease,
        x='occ_title',
        y='percent_change',
        title='Top 10 Jobs with Highest Employment Decrease',
        labels={'percent_change': '% Decrease'},
        color_discrete_sequence=['red'],
        text='percent_change'
    )
    fig_decrease.update_layout(height=600, width=800,
                               xaxis_tickangle=-45, margin={"t": 50, "l": 50, "r": 50, "b": 150})
    fig_decrease.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    return fig_increase, fig_decrease, result_widget_options

if __name__ == '__main__':
    # use_reloader=False to prevent restarting
    work_trends_2029.run_server(debug=True, port=8055, use_reloader=False)
