"""
employees_across_sectors: Geographic Distribution of Employees Across Different Industries

This module contains functions for an app to show geoplot of employees across industries .
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import geopandas as gpd
from data_preprocessing import load_and_process_payroll_data

usa_gdf = gpd.read_file("project_data/cb_2018_us_state_5m/cb_2018_us_state_5m.shp")
usa_gdf = usa_gdf.rename(columns={'NAME': 'State'})
payroll_df = load_and_process_payroll_data()
df_cpy = payroll_df.copy()

merged_df = usa_gdf.merge(df_cpy, on='State')

employees_across_sectors = dash.Dash(__name__)

employees_across_sectors.layout = html.Div([
    dcc.Dropdown(
        id='occupation-dropdown',
        options=[{'label': i, 'value': i} for i in df_cpy.columns[2:]],
        value='Construction'
    ),
    dcc.Graph(id='map-graph')
])

@employees_across_sectors.callback(
    Output('map-graph', 'figure'),
    Input('occupation-dropdown', 'value')
)
def update_graph(selected_occupation):
    """
    Update the choropleth map based on the selected occupation.

    Args:
        selected_occupation (str): The selected occupation whose data will be displayed on the map.

    Returns:
        plotly.graph_objects.Figure: The updated figure for the map displaying
        the selected occupation's data.
    """
    occupation_values = merged_df[selected_occupation]
    min_value = occupation_values.min()
    max_value = occupation_values.max()

    fig = px.choropleth_mapbox(
        merged_df,
        geojson=merged_df.geometry,
        locations=merged_df.index,
        color=selected_occupation,
        color_continuous_scale="Turbo",
        range_color=[min_value, max_value],  # Dynamic scale
        mapbox_style="carto-positron",
        zoom=1.75,
        center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.7,
        labels={selected_occupation: 'Number of Employees'}
    )

    fig.update_layout(
        title=f"Number of Employees by State for {selected_occupation}",
        title_x=0.5,  # Center the title
        title_font={"size": 20},  # Set the font size
        margin={"r": 0, "t": 40, "l": 0, "b": 0}  # Adjust top margin for title
    )

    return fig

if __name__ == '__main__':
    employees_across_sectors.run_server(debug=True, port=8058)
