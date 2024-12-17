"""
data_preprocessing: Data cleaning and formatting for use in project.

This module contains functions clean different datasets and format them for visualisation .
"""
from pathlib import Path
import pandas as pd
import numpy as np

def load_and_clean_skills_occupation_data():
    """
    Loads and cleans occupation and skills data for 2023 and 2029.

    This function reads and cleans occupation data for the years 2023 and 2029,
    along with skills data for 2023. It merges the occupation data on a common
    occupation code and calculates changes in employment between 2023 and 2029. 
    The skills data is loaded from a specific sheet in an Excel file, and the data
    frames are cleaned and prepared for analysis.
    The function assumes that the required Excel files are present in the
    'project_data' folder within the current working directory.

    Returns:
        tuple:
            - common_occupation_df (pd.DataFrame): Merged and cleaned occupation data 
              for comparison, with columns such as employment figures and occupational openings.

            - skills_6_1 (pd.DataFrame): Merged and cleaned skills data from the 'Table 6.1' sheet.

    Raises:
        FileNotFoundError: If the required Excel files are not found in the expected paths.
        ValueError: If there are issues with the structure or content of the data during cleaning.

    Example:
        common_occupation_df, skills_6_1 = load_and_clean_skills_occupation_data()
        print(common_occupation_df.head())
        print(skills_6_1.head())
    """
    cwd = Path.cwd()
    skills_path = (cwd/'project_data'/'2023-33'/'skills.xlsx').resolve()
    skills_6_1 = pd.read_excel(skills_path, sheet_name="Table 6.1", skiprows=1, header=0)
    skills_6_1['2023 National Employment Matrix code'] = \
        skills_6_1['2023 National Employment Matrix code'].str.split('-').str[0]

    skills_6_1.drop(skills_6_1.tail(4).index, inplace=True)
    skills_6_1.iloc[:,0] = skills_6_1.iloc[:,0].apply(lambda x: x.strip())

    occ_2029_path = 'project_data/2019-29/occupation.xlsx'
    occupation_df_2029 = pd.read_excel(occ_2029_path,sheet_name="Table 1.2",skiprows=1,header=0)
    occ_2023_path = 'project_data/2023-33/occupation.xlsx'
    occupation_df_2023 = pd.read_excel(occ_2023_path, sheet_name="Table 1.2", skiprows=1, header=0)

    occupation_df_2023.drop(occupation_df_2023.tail(4).index,inplace=True)
    occ_rename_dict = {'2019 National Employment Matrix title': 'occ_title',\
                       '2019 National Employment Matrix code': 'occ_code'}
    occupation_2029_renamed = occupation_df_2029.rename(columns=occ_rename_dict)
    occ_rename_dict_23 = {'2023 National Employment Matrix title': 'occ_title',\
                           '2023 National Employment Matrix code': 'occ_code'}
    occupation_2023_renamed = occupation_df_2023.rename(columns=occ_rename_dict_23)

    columns_extract_29 = ['occ_title', 'occ_code', 'Employment, 2029']
    occupation_2029_renamed = occupation_2029_renamed[columns_extract_29]
    columns_extract_23 = ['occ_title', 'occ_code', 'Employment, 2023']
    occupation_2023_renamed = occupation_2023_renamed[columns_extract_23]

    common_occupation_df = pd.merge(occupation_2029_renamed, \
                                    occupation_2023_renamed, on=['occ_code'], how='inner')
    common_occupation_df.drop('occ_title_y',inplace=True, axis=1)
    common_occupation_df.rename(columns={'occ_title_x':'occ_title'}, inplace=True)

    common_occupation_df['change'] = ((common_occupation_df['Employment, 2029'] -\
                                        common_occupation_df['Employment, 2023']))
    common_occupation_df['percent_change'] = ((common_occupation_df['change']) / \
                                              common_occupation_df['Employment, 2023']) * 100
    common_occupation_df = common_occupation_df.iloc[1:,:]
    common_occupation_df = \
        common_occupation_df[~common_occupation_df['occ_code'].str.contains('-0000')]

    return common_occupation_df, skills_6_1

def load_and_clean_skills_6_2_data():
    """
    Loads and cleans the Skills 6.2 data from an Excel file.

    This function reads the "Table 6.2" sheet from the Excel file located in the 
    'project_data/2023-33' directory. It performs the following operations:
    1. Skips the first row and uses the second row as the header.
    2. Drops the last 6 rows of the dataset.
    3. Removes the first row of the dataset and resets the index.
    4. Splits the '2023 National Employment Matrix code' column by the
    hyphen ('-') and keeps only the first part of the code.

    Returns:
        pandas.DataFrame: The cleaned DataFrame containing the data from 
        the "Table 6.2" sheet with the appropriate transformations applied.

    Example:
        >>> df = load_and_clean_skills_6_2_data()
        >>> df.head()
    """
    cwd = Path.cwd()
    skills_6_2_path = (cwd/'project_data'/'2023-33'/'skills.xlsx').resolve()
    skills_6_2 = pd.read_excel(skills_6_2_path, sheet_name="Table 6.2", skiprows=1, header=0)
    skills_6_2 = skills_6_2.iloc[:-6]
    skills_6_2 = skills_6_2.drop(index=0).reset_index(drop=True)
    skills_6_2['2023 National Employment Matrix code'] = \
        skills_6_2['2023 National Employment Matrix code'].str.split('-').str[0]
    return skills_6_2

def load_and_clean_wage_data(wage_data_filepath):
    """
    Load and clean wage data from an Excel file.

    This function reads an Excel file containing wage data, cleans the data by 
    removing unnecessary columns, replacing special characters in specific columns,
    and converting the columns to numeric types. It also drops duplicates based on 
    the 'OCC_TITLE' column.

    Args:
        wage_data_filepath (str or Path): The path to the Excel file containing wage data.

    Returns:
        pandas.DataFrame: A cleaned DataFrame with selected columns from the wage data.

    Raises:
        FileNotFoundError: If the file at the given `wage_data_filepath` is not found.
        ValueError: If there are issues with data conversion during cleaning.

    Example:
        wage_data = load_and_clean_wage_data('project_data/oesm23nat/national_M2023_dl.xlsx')
        print(wage_data.head())    
    """
    cwd = Path.cwd()
    # wage_2023 = pd.read_excel((cwd/'project_data/oesm23nat'/"national_M2023_dl.xlsx").resolve())
    wage_2023 = pd.read_excel((cwd/wage_data_filepath).resolve())
    drop_columns = ['AREA','AREA_TITLE','AREA_TYPE','NAICS','NAICS_TITLE',
                    'I_GROUP','JOBS_1000', 'LOC_QUOTIENT', 'PCT_TOTAL','PCT_RPT']
    wage_2023.drop(drop_columns, axis=1, inplace=True)

    columns_to_replace = [
        'H_MEAN','H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90',
        'A_MEAN','A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90'
    ]
    wage_2023[columns_to_replace] = wage_2023[columns_to_replace].astype(str)

    wage_2023[columns_to_replace] = wage_2023[columns_to_replace].map(
            lambda x: x.replace('*', '0').replace('#', '0')
    )

    wage_2023[columns_to_replace] = wage_2023[columns_to_replace].astype(np.float64)
    wage_2023['OCC_CODE'] = wage_2023['OCC_CODE'].astype('str')
    wage_2023['OCC_CODE'] = wage_2023['OCC_CODE'].str.split('-').str[0]
    wage_2023.drop_duplicates(['OCC_TITLE'],inplace=True)

    return wage_2023

def load_and_process_payroll_data():
    """
    Load and process the employee payroll data from an Excel file.

    This function reads payroll data from the "Employee_NonfarmPayrolls.xlsx" file, which is located 
    in the "project_data" directory. The data is processed by cleaning the numerical columns
    (removing commas and converting to float), then multiplying the values by 1000.

    Returns:
        pd.DataFrame: The processed payroll DataFrame with cleaned numerical values.

    Raises:
        FileNotFoundError: If the Excel file does not exist at the specified path.
        ValueError: If there is an issue with the data conversion process.
    """
    cwd = Path.cwd()
    path = cwd / "project_data" / "Employee_NonfarmPayrolls.xlsx"
    payroll_data_path = path.resolve()

    payroll_df = pd.read_excel(payroll_data_path, sheet_name=5)

    payroll_numerical_cols = [
        'Total', 'Mining and logging', 'Construction', 'Manufacturing',
        'Trade, transportation, and utilities', 'Information',
        'Financial activities', 'Professional and business services',
        'Education and health services', 'Leisure and hospitality',
        'Other services', 'Government'
    ]

    try:
        payroll_df[payroll_numerical_cols] = payroll_df[payroll_numerical_cols].astype(str)
        payroll_df[payroll_numerical_cols] = payroll_df[payroll_numerical_cols].map(
            lambda x: x.replace(",", "")
        )
        payroll_df[payroll_numerical_cols] = payroll_df[payroll_numerical_cols].astype(float)
        payroll_df[payroll_numerical_cols] = payroll_df[payroll_numerical_cols] * 1000
    except Exception as e:
        raise ValueError(f"Error processing the numerical columns: {e}") from e

    return payroll_df

def process_wage_and_employment_data():
    """
    Processes national wage and employment data from 2019 and 2023, merges the data, 
    and returns a DataFrame with calculated employment changes and department information.

    This function performs the following operations:
    1. Reads the 2019 and 2023 national wage and employment data from Excel files.
    2. Merges the data for both years based on occupation code and title.
    3. Renames the columns for clarity.
    4. Calculates the total employment change and the percentage change.
    5. Extracts department information and adds the department number and name to the dataset.
    6. Cleans the data by removing rows with certain patterns.

    Returns:
        pd.DataFrame: A DataFrame containing merged and cleaned wage, employment data, 
                      with employment changes and department information.
    """
    nat_wage_2019 = pd.read_excel('project_data/oesm19nat/national_M2019_dl.xlsx')
    columns_to_extract = ['occ_code', 'occ_title', 'tot_emp', 'a_mean']
    nat_wage_2019 = nat_wage_2019[columns_to_extract]

    nat_wage_2023 = pd.read_excel('project_data/oesm23nat/national_M2023_dl.xlsx')
    nat_wage_2023.columns = nat_wage_2023.columns.str.lower()
    nat_wage_2023 = nat_wage_2023[columns_to_extract]

    merged_df = pd.merge(nat_wage_2019[columns_to_extract],
                         nat_wage_2023[columns_to_extract],
                         on=['occ_code', 'occ_title'], how='outer',
                              suffixes=('_nat_wage_2019', '_nat_wage_2023'))

    merged_df.rename(columns={
        'tot_emp_nat_wage_2019': 'tot_emp_2019', 
        'tot_emp_nat_wage_2023': 'tot_emp_2023',
        'a_mean_nat_wage_2019': 'annual_2019', 
        'a_mean_nat_wage_2023': 'annual_2023'
    }, inplace=True)

    new_column_order = ["occ_code", "occ_title", "tot_emp_2019",\
                         "tot_emp_2023", "annual_2019", "annual_2023"]
    merged_df = merged_df[new_column_order]

    common_df = pd.merge(nat_wage_2019[columns_to_extract],
                          nat_wage_2023[columns_to_extract],
                         on=['occ_code', 'occ_title'], how='inner',
                           suffixes=('_nat_wage_2019', '_nat_wage_2023'))

    common_df.rename(columns={
        'tot_emp_nat_wage_2019': 'tot_emp_2019', 
        'tot_emp_nat_wage_2023': 'tot_emp_2023',
        'a_mean_nat_wage_2019': 'annual_2019', 
        'a_mean_nat_wage_2023': 'annual_2023'
    }, inplace=True)

    common_df['change'] = common_df['tot_emp_2023'] - common_df['tot_emp_2019']
    common_df['percent_change'] = (common_df['change'] / common_df['tot_emp_2019']) * 100

    department_df = pd.read_excel('project_data/2023-33/skills.xlsx', sheet_name=1, skiprows=1)
    department_df = department_df[:23]

    departments = {row['2023 National Employment Matrix title']:
                   row['2023 National Employment Matrix code'].split('-')[0]
                   for _, row in department_df.iterrows()}
    del departments['Total, all occupations']
    departments = {key.strip(): value for key, value in departments.items()}

    common_df['department_number'] = common_df['occ_code'].str.split('-').str[0].astype(str)
    common_df['department_name'] = common_df['department_number'].map(
        {v: k for k, v in departments.items()})

    common_df = common_df[1:]
    common_df = common_df[~common_df['occ_code'].str.contains('-0000')]

    return common_df
