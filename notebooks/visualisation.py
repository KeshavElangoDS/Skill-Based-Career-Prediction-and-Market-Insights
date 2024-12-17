"""
visualisation: Functions to display various dynamic plots.
This module contains functions to show interactive visualizations.
"""
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import panel as pn

import numpy as np
import ipywidgets as widgets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from ipywidgets import VBox, HBox
from IPython.display import display

from data_preprocessing import load_and_clean_wage_data, load_and_clean_skills_occupation_data

def get_top_5_matrix_codes(df, top_5_label_global):
    """ 
    Retrieves the matrix codes for the top 5 job titles.

    Args:
        df (pd.DataFrame): DataFrame containing job matrix data.
        top_5_labels_global (list): List of top 5 job titles.

    Returns:
        list: A list of the corresponding matrix codes for the selected job titles.
    """
    top_5_codes = []

    for title in top_5_label_global:
        code = df[df['2023 National Employment Matrix title'] == title]\
            ['2023 National Employment Matrix code'].str.strip().values
        if len(code) > 0:
            top_5_codes.append(code[0])

    return top_5_codes


def plot_wage_vs_employment(df, matrix_codes, job_titles):
    """
    Generates a scatter plot comparing wages and total employment for selected job titles.
    The data used is the `national_M2023_dl.xlsx` under project_data folder.
    The path of the data should be '.project_data/oesm23nat/national_M2023_dl.xlsx'

    Args:
        df (DataFrame): Dataframe containing the occupation title and occupation code.
        matrix_codes (list): List of matrix codes corresponding to job titles.
        job_titles (list): List of job titles for comparison.

    Returns:
        pn.pane.Plotly: A Plotly figure wrapped in Panel for display.
    """
    cwd = Path.cwd()
    data = pd.read_excel((cwd/'project_data/oesm23nat/national_M2023_dl.xlsx').resolve())

    filtered_data = data[data['OCC_CODE'].str.split('-').str[0].isin(matrix_codes)]

    job_title_lower = [jb.lower() for jb in job_titles]
    filtered_data = filtered_data[~filtered_data['OCC_TITLE'].str.lower().isin(job_title_lower)]

    if filtered_data.empty:
        return pn.pane.Markdown("No data found for the selected matrix codes,\
                                 excluding selected titles.")

    filtered_data['OCC_PREFIX'] = filtered_data['OCC_CODE'].str.slice(0, 2)

    color_map = {prefix: px.colors.qualitative.Set1\
                 [i % len(px.colors.qualitative.Set1)] \
                    for i, prefix in enumerate(filtered_data['OCC_PREFIX'].unique())}
    filtered_data['Color'] = filtered_data['OCC_PREFIX'].map(color_map)

    occ_prefix_to_titles = {}
    for title in job_titles:
        matrix_code = df[df['2023 National Employment Matrix title'].str.strip() == title]\
            ['2023 National Employment Matrix code'].values
        if len(matrix_code) > 0:
            prefix = matrix_code[0][:2]  # Get the first two digits of the matrix code (OCC_PREFIX)
            occ_prefix_to_titles[prefix] = title  # Map OCC_PREFIX to the selected job title

    fig = px.scatter(
        filtered_data,
        x='TOT_EMP',
        y='A_MEAN',
        hover_data=['OCC_TITLE'],
        title="Wage vs. Total Employment for Selected Job Groups",
        labels={'TOT_EMP': 'Total Employment', 'A_MEAN': 'Average Mean Wage'},
        color='OCC_PREFIX',
        color_discrete_map=color_map,
        width=1500,
        height=400
    )

    fig.for_each_trace(lambda t: t.update(name=occ_prefix_to_titles.get(t.name, t.name)))

    return pn.pane.Plotly(fig, height=500)

def create_interactive_wage_plot(df, result_widget_name):
    """
    Creates an interactive wage plot based on selected job groups.

    Args:
        df (pd.DataFrame): DataFrame containing job matrix data.
        result_widget_name (list): List of job titles for interactive selection.

    Returns:
        pn.Column: A Panel layout containing the interactive widgets and plot.
    """
    result_widget = pn.widgets.MultiSelect(name="Select Job Groups",
                                           options=list(result_widget_name),
                                           value=list(result_widget_name),
                                           height=110, width=450)
    result_widget.margin = (0, 0, 30, 0)

    def plot_wage_data(result):
        """Wrapper function to generate wage vs. employment plot based on selected job groups."""
        selected_titles = result
        matrix_codes = df[df['2023 National Employment Matrix title']\
                          .str.strip().isin(selected_titles)]\
                            ['2023 National Employment Matrix code'].values

        if len(matrix_codes) == 0:
            return pn.pane.Markdown("No valid job titles selected or matching.")

        return plot_wage_vs_employment(df, matrix_codes, selected_titles)

    interaction = pn.interact(plot_wage_data, result=result_widget)

    return pn.Column(
        pn.Row(pn.pane.Markdown(\
            "## Top Job Groups by 2023 Annual Median Wage and Total Employment"), align='center'),
        interaction[0],
        interaction[1]
    ).servable()

def plot_wage_data_by_prefix(result):
    """
    Retrieves the job codes for the selected prefixes, filters the wage data 
    based on those job codes,and generates a Plotly bar chart for the top 10 
    occupations based on their average median wage.

    This function performs the following steps:
    1. Filters the job codes based on the selected job prefixes.
    2. Filters the wage data based on the retrieved job codes and sorts it by average median wage.
    3. Creates a Plotly bar chart for the top 10 occupations based on their average median wage.

    Args:
        result (list): A list of job prefix strings selected by the user (e.g., ['15', '17']).

    Returns:
        pn.pane.Plotly or pn.pane.Markdown: 
            - A Plotly chart displaying the top 10 jobs with the highest average median wage.
            - A Markdown message indicating that no data is available if no matching 
            job codes are found.
    """

    _, skills_6_1 = load_and_clean_skills_occupation_data()
    wage_data_filepath = "project_data/oesm23nat/national_M2023_dl.xlsx"
    wage_2023 = load_and_clean_wage_data(wage_data_filepath)

    job_codes = skills_6_1[skills_6_1['2023 National Employment Matrix title']
                   .str.strip()
                   .str.startswith(tuple(result))]['2023 National Employment Matrix code'].values

    mask = wage_2023.OCC_CODE.str.startswith(tuple(job_codes))
    filtered_data = wage_2023[mask].sort_values(['A_MEAN'], ascending=False)

    if filtered_data.empty:
        return pn.pane.Markdown(\
            "No data available for the selected job prefixes.")

    top_10_data = filtered_data.head(10)

    fig = px.bar(
        top_10_data,
        x='A_MEAN',
        y='OCC_TITLE',
        title="Top 10 Jobs by 2023 Annual Median Wage",
        labels={'OCC_TITLE': 'Occupation Title', 'A_MEAN': 'Average Median Wage'},
        color='A_MEAN',
        color_continuous_scale='Viridis',
        height=400
    )

    fig.update_layout(
        yaxis_title='Occupation Title',
        xaxis_title='Average Median Wage',
        showlegend=False
    )

    return pn.pane.Plotly(fig, height=400, width=1500)

def create_interactive_job_prefix_plot(skills_6_1, top_5_codes):
    """
    Creates an interactive plot for selecting job prefixes and displaying corresponding wage data.

    This function generates a MultiSelect widget based on the `top_5_codes` list, which contains
    the prefixes of the 2023 National Employment Matrix codes. It then filters the `skills_6_1`
    DataFrame based on these prefixes to show the corresponding job titles.
    The selected job prefixes used for a plot displaying the top 10 jobs by median wage.

    Args:
        skills_6_1 (pd.DataFrame): A DataFrame containing skills and occupation data, including 
        job titles and corresponding matrix codes.

        top_5_codes (list of str): A list of the top 5 matrix code prefixes used to filter
        job titles.

    Returns:
        pn.Column: A Panel layout containing the interactive widgets and plot.

    Example:
        skills_6_1 = pd.read_excel("path_to_skills_data.xlsx")
        top_5_codes = ['15', '17', '11', '13', '23']
        plot = create_interactive_job_prefix_plot(skills_6_1, top_5_codes)
        plot.servable()
    """
    _, skills_6_1 = load_and_clean_skills_occupation_data()

    occ_code_mask = skills_6_1['2023 National Employment Matrix code']\
        .str.startswith(tuple(top_5_codes))
    job_prefix_names = skills_6_1[occ_code_mask]['2023 National Employment Matrix title']\
        .str.strip().values

    result_widget = pn.widgets.MultiSelect(
        name="Select Job Prefixes",
        options=list(job_prefix_names),
        value=list(job_prefix_names),  # Default to all selected
        height=120,
        width=450
    )
    result_widget.margin = (0, 0, 30, 0)

    interaction = pn.interact(plot_wage_data_by_prefix, result=result_widget)

    return pn.Column(
        pn.Row(pn.pane.Markdown("## Top 10 Jobs by 2023 Annual Median Wage Data"), align='center'),
        interaction,
    )

# Global variables to store results
top_5_labels_global = []
top_5_probs_global = []

def riasec_prediction_model(df):
    """
    Trains a RIASEC-based job prediction model and allows the user to input their 
    RIASEC scores to get job predictions.

    Args:
        df (pd.DataFrame): A DataFrame containing the necessary features and target for 
                           training the model. The DataFrame should have columns for RIASEC 
                           attributes, such as: 'Mechanical', 'Physical strength and stamina', 
                           'Fine motor', etc. Additionally, it must contain 
                           '2023 National Employment Matrix code' 
                           and '2023 National Employment Matrix title'.
    
    Returns:
        None: This function displays the user interface and updates the global variables 
        `top_5_labels_global`.
    """

    riasec_mapping = {
        'Realistic': ['Mechanical', 'Physical strength and stamina', 'Fine motor'],
        'Investigative': ['Science', 'Mathematics', 'Critical and analytical thinking'],
        'Artistic': ['Creativity and innovation', 'Writing and reading'],
        'Social': ['Interpersonal', 'Speaking and listening', 'Customer service'],
        'Enterprising': ['Leadership', 'Project management', 
                         'Problem solving and decision making', 'Adaptability'],
        'Conventional': ['Computers and information technology', 'Detail oriented', 'Adaptability'],
    }

    for theme, columns in riasec_mapping.items():
        df[theme] = df[columns].mean(axis=1)

    x = df[['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']]
    x = x.fillna(x.mean())

    y = df['2023 National Employment Matrix code']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    titles = df['2023 National Employment Matrix title']
    _ = label_encoder.fit_transform(titles)

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.15, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # -----------------------
    # Widgets for User Input
    # -----------------------
    widgets_dict = {
        'Realistic': widgets.FloatText(description='Realistic', value=0),
        'Investigative': widgets.FloatText(description='Investigative', value=0),
        'Artistic': widgets.FloatText(description='Artistic', value=0),
        'Social': widgets.FloatText(description='Social', value=0),
        'Enterprising': widgets.FloatText(description='Enterprising', value=0),
        'Conventional': widgets.FloatText(description='Conventional', value=0)
    }

    user_inputs = {}
    x_new = None

    def get_user_input():
        """
        Collects user input from the RIASEC widgets and stores it for prediction.

        Returns:
            pd.DataFrame: The user input formatted into a dataframe for prediction.
        """
        global x_new
        for feature, widget in widgets_dict.items():
            user_inputs[feature] = widget.value

        x_new = pd.DataFrame([user_inputs])
        return x_new

    print("\t\t\t\t\t\t\t RIASEC Mapping \t\t\t\t\t\t\t")

    # Submit button callback
    def on_submit(b):
        """
        Handles the submit button click event. Collects user input, makes predictions,
        and stores the results.

        Updates the `top_5_labels_global` and `top_5_probs_global` for prediction results.
        """
        global top_5_labels_global, top_5_probs_global

        # Get user inputs
        x_new = get_user_input()

        if x_new is not None:
            y_new_pred_prob = model.predict_proba(x_new)
            top_5_indices = np.argsort(y_new_pred_prob[0])[-5:][::-1]

            top_5_labels_global = label_encoder.inverse_transform(top_5_indices)
            top_5_probs_global = y_new_pred_prob[0][top_5_indices]

            top_5_labels_global = list(top_5_labels_global)

            print("\nTop 5 predicted target labels for the new data:")
            for idx, label in enumerate(top_5_labels_global):
                print(f"{idx + 1}: {label} (Probability: {top_5_probs_global[idx]:.4f})")

            predicted_title = top_5_labels_global[0]
            print(f"\nMost likely job title: {predicted_title}")

            plot_top_5_confidence(top_5_labels_global, top_5_probs_global)

    submit_button = widgets.Button(description="Submit")
    submit_button.on_click(on_submit)

    display_widgets = []
    for _, widget in widgets_dict.items():
        widget.layout.width = 'auto'
        display_widgets.append(HBox([widget]))

    display(VBox(display_widgets))
    display(submit_button)

    print("Waiting for user input...")

    def plot_top_5_confidence(labels, probabilities):
        """
        Plots a bar chart showing the top 5 predicted labels with their confidence levels.
        
        Args:
            labels (list): A list of top 5 predicted labels.
            probabilities (list): Corresponding probabilities of the predicted labels.
        """
        plt.figure(figsize=(8, 4))
        sns.barplot(y=labels, x=probabilities, hue=labels, palette='viridis')
        plt.title('Top 5 Predicted Labels and Their Confidence')
        plt.ylabel('Labels')
        plt.xlabel('Confidence (Probability)')
        plt.xticks(rotation=45, ha='right')
        plt.show()


# Helper function to get results from global variable after submission
def get_top_5_predictions():
    """
    Returns the top 5 predicted job titles after user interaction.
    
    Returns:
        list: A list of top 5 predicted job titles.
    """
    return top_5_labels_global


def plot_education_distribution(df1, df_selected):
    """
    Function to generate an interactive plot showing the distribution of people by education level 
    for a selected occupation title, based on the 2023 National Employment Matrix.

    This function processes the input data, groups it by occupation code and education level, 
    and provides a pie chart visualization of the total number of people by education level for 
    the selected occupation.

    Args:
        df1 (pd.DataFrame): DataFrame containing the employment data, with columns:
                             - "2023 National Employment Matrix code"
                             - "Typical education needed for entry"
                             - "Employment, 2023"
        df_selected (pd.DataFrame): DataFrame containing job titles, with columns:
                                    - "2023 National Employment Matrix code"
                                    - "2023 National Employment Matrix title"

    Returns:
        panel.pane.Markdown or panel.pane.Plotly: A markdown message or an interactive pie chart
                                                  depending on the occupation title selected.
    """

    grouped = df1.groupby(["2023 National Employment Matrix code", \
                           "Typical education needed for entry"])\
                            ["Employment, 2023"].sum().reset_index()

    pivot_table = grouped.pivot(index="2023 National Employment Matrix code",
                                columns="Typical education needed for entry",
                                values="Employment, 2023").fillna(0)

    pivot_table.columns.name = None
    pivot_table.reset_index(inplace=True)

    education_df = pivot_table
    df_melted = education_df.melt(id_vars=["2023 National Employment Matrix code"],
                                  var_name="Education Level",
                                  value_name="Number of People")

    def plot_people_data(occupation_title):
        """
        Helper function to plot the pie chart for the selected occupation title.
        
        Args:
            occupation_title (str): The occupation title for which to generate the plot.

        Returns:
            pn.pane.Plotly or pn.pane.Markdown: The Plotly pie chart or a markdown message if 
                                                 no data is available.
        """

        occupation_code = df_selected[df_selected\
                                    ['2023 National Employment Matrix title'] == occupation_title]\
                                    ['2023 National Employment Matrix code'].iloc[0]

        filtered_data = df_melted[df_melted['2023 National Employment Matrix code']\
                                   == occupation_code]

        if filtered_data.empty:
            return pn.pane.Markdown("No data available for the selected occupation code.")

        summed_data = filtered_data.groupby('Education Level')\
            ['Number of People'].sum().reset_index()

        fig = px.pie(
            summed_data,
            names='Education Level',
            values='Number of People',
            title=f"Total People by Education Level for {occupation_title}",
            color='Education Level',
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        fig.update_layout(
            showlegend=True,
            xaxis_title="Education Level",
            yaxis_title="Number of People",
        )

        return pn.pane.Plotly(fig, height=600, width=900)

    occupation_title_widget = pn.widgets.Select(
        name="Select Occupation Title",
        options=list(
            df_selected["2023 National Employment Matrix title"].unique()),  # Use titles as options
        value=df_selected["2023 National Employment Matrix title"].iloc[0],  # Default value
        height=50
    )

    occupation_title_widget.margin = (0, 0, 30, 0)

    interaction = pn.interact(plot_people_data, occupation_title=occupation_title_widget)

    return pn.Column(
        pn.Row(pn.pane.Markdown("## Total Number of People by Education Level\
                                 for Selected Occupation - 2023"), align='center'),
        interaction,
    ).servable()
