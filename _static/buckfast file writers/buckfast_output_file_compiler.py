'''
Script to convert the Buckfast output file into the BuckPy output file format
'''

import numpy as np
import pandas as pd
import pandas.io.formats.excel
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

def find_buckfast_line_no():

    """
    Find the line number of the df from the Buckfast result file.

    Parameters
    ----------
    None

    Returns
    -------
    line_elements_start : int
        The line number of the start of the element with a buckle in the 'Sets' Tab.
    line_elements_end : int
        The line number of the end of the element with a buckle in the 'Sets' Tab.
    line_set_prob_start : int
        The line number of the start of the element set probability in the 'Sets' Tab.
    line_set_prob_end : int
        The line number of the end of the element set probability in the 'Sets' Tab.
    line_no_buckle_start : int
        The line number of the start of the number of buckles in the 'No Buckles' Tab.
    line_no_buckle_end : int
        The line number of the end of the number of buckles in the 'No Buckles' Tab.
    line_pipe_start : int
        The line number of the start of the pipeline data for the analytical EAF.
    line_pipe_end : int
        The line number of the end of the pipeline data for the analytical EAF.
    """

    # Read Buckfast result file '.out2'
    count = 0
    line_elements_start, line_set_prob_start, line_set_prob_end, line_no_buckle_start = 0, 0, 0, 0
    line_elset_list = []
    with open(f"{WORK_DIR}\\buckfast_{PIPELINE_ID}_scen{SCENARIO}.out2.csv") as f:
        for line in f:
            if 'centroid of element' in line and 'number of simulations with a buckle' in line:
                line_elements_start = count
            if 'element set' in line and 'number of simulations with a buckle' in line:
                line_set_prob_start = count
            if 'VAS' in line and 'frequency' in line:
                line_set_prob_end = count
            if 'number of buckles' in line and 'frequency' in line:
                line_no_buckle_start = count
            if '*ELSET' in line:
                line_elset_list.append(count)
            count += 1
    line_elements_end = line_no_buckle_start
    line_no_buckle_end = line_elset_list[0]

    # Read Buckfast result file '.out1'
    count = 0
    line_force_start, line_force_end, line_pipe_start, line_pipe_end = 0, 0, 0, 0
    with open(f"{WORK_DIR}\\buckfast_{PIPELINE_ID}_scen{SCENARIO}.out1.csv") as f:
        for line in f:
            if 'centroid of element' in line and 'type of element' in line:
                line_force_start = count
            if 'centroid of element' in line and 'force profile during installation' in line:
                line_force_end = count
            if '*SINGLE' in line:
                line_pipe_start = count
            if 'density of contents during installation (kg.m-3)' in line:
                line_pipe_end = count
            count += 1

    return line_elements_start, line_elements_end, line_set_prob_start, line_set_prob_end,\
           line_no_buckle_start, line_no_buckle_end, line_force_start, line_force_end,\
           line_pipe_start, line_pipe_end

def create_buckfast_tab_elements(line_start, line_end):

    """
    Create the dataframe of the 'Elements' Tab in BuckPy results.

    Parameters
    ----------
    line_start : int
        The line number of the start of the element with a buckle in the 'Elements' Tab.
    line_end : int
        The line number of the end of the element with a buckle in the 'Elements' Tab.

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing the data related to the element with a buckle.
    """

    # Select rows and columns from the Buckfast result and convert data type to float
    cols = ['centroid of element', 'number of simulations with a buckle',
            'probability of buckling', 'probability of not buckling', 'mean of the VAS',
            'standard deviation of the VAS', 'minimum VAS', 'maximum VAS']
    df = pd.read_csv(f"{WORK_DIR}\\buckfast_{PIPELINE_ID}_scen{SCENARIO}.out2.csv",
                     encoding = 'utf-8', skiprows = line_start, nrows = line_end - line_start - 2,
                     usecols = cols).dropna().astype(float)

    # Rename the column names
    cols = ['Centroid of the Element (m)', 'Number of Simulations with a Buckle',
            'Probability of Buckling', 'Probability of not Buckling', 'Mean of the VAS (m)',
            'Standard Deviation of the VAS (m)', 'Minimum VAS (m)', 'Maximum VAS (m)']
    df.columns = cols

    return df

def create_buckfast_tab_sets(line_start, line_end):

    """
    Create the dataframe of the 'Sets' Tab in BuckPy results.

    Parameters
    ----------
    line_start : int
        The line number of the start of the element set probability in the 'Sets' Tab.
    line_end : int
        The line number of the end of the element set probability in the 'Sets' Tab.

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing the data related to the element set probability.
    """

    # Select rows and columns from the Buckfast result and convert data type to float
    cols = ['element set', 'number of simulations with a buckle', 'probability of buckling',
            'probability of not buckling', 'mean of the VAS', 'standard deviation of the VAS',
            'minimum VAS', 'maximum VAS', 'conditional VAS', 'unconditional VAS',
            'conditional VAS.1', 'unconditional VAS.1']
    df = pd.read_csv(f"{WORK_DIR}\\buckfast_{PIPELINE_ID}_scen{SCENARIO}.out2.csv",
                     encoding = 'utf-8', skiprows = line_start,
                     nrows = line_end - line_start - 4, usecols = cols).dropna().astype(float)

    # Rename the column names
    cols = ['Set Label', 'Number of Simulations with Buckles per Set', 'Probability of Buckling',
            'Probability of not Buckling', 'Mean of the VAS (m)',
            'Standard Deviation of the VAS (m)', 'Minimum VAS (m)', 'Maximum VAS (m)',
            'VAS, Conditional, Rogue (m)', 'Characteristic VAS, Unconditional, Rogue (m)',
            'VAS, Conditional, Planned (m)', 'Characteristic VAS, Unconditional, Planned (m)']
    df.columns = cols

    return df

def create_buckfast_tab_no_buckles(line_start, line_end):

    """
    Create the dataframe of the 'No Buckles' Tab in BuckPy results.

    Parameters
    ----------
    line_start : int
        The line number of the start of the number of buckles in the 'No Buckles' Tab.
    line_end : int
        The line number of the end of the number of buckles in the 'No Buckles' Tab.

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing the data related to the number of buckles.
    """

    # Select rows and columns from the Buckfast result and convert data type to float
    cols = ['number of buckles', 'frequency', 'probability', 'cumulative probability']
    df = pd.read_csv(f"{WORK_DIR}\\buckfast_{PIPELINE_ID}_scen{SCENARIO}.out2.csv",
                     encoding = 'utf-8', skiprows = line_start,
                     nrows = line_end - line_start - 3, usecols = cols).dropna().astype(float)

    # Rename the column names
    cols = ['Number of Buckles', 'Number of Simulations', 'Probability of Buckling',
            'Cumulative Probability of Buckling']
    df.columns = cols

    return df

def create_buckfast_tab_force_prof(line_start, line_end):

    """
    Create the dataframe of the 'Force Profiles' Tab in BuckPy results.

    Parameters
    ----------
    line_start : int
        The line number of the start of the force profile data in the 'Force Profiles' Tab.
    line_end : int
        The line number of the end of the force profile data in the 'Force Profiles' Tab.

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing the data related to the force profile data.
    """

    # Select the CBF columns in the Buckfast result
    cols = ['centroid of element', 'mean', 'mean.1']
    df_cbf = pd.read_csv(f"{WORK_DIR}\\buckfast_{PIPELINE_ID}_scen{SCENARIO}.out1.csv",
                     encoding = 'utf-8', skiprows = line_start,
                     nrows = line_end - line_start - 3, usecols = cols)
    df_cbf = df_cbf.iloc[1:,:].reset_index(drop = True).astype(float)

    # Rename the column names and convert unit to kN
    cols = ['KP (m)', 'CBF Hydrotest (kN)', 'CBF Operation (kN)']
    df_cbf.columns = cols
    df_cbf.loc[:, cols[1:]] = df_cbf.loc[:, cols[1:]] / 1000.0

    # Select the EAF columns in the Buckfast result
    cols = ['centroid of element', 'force profile during installation',
            'force profile during hydrotest', 'fully developed force profile',
            'force profile during operation']
    df_eaf = pd.read_csv(f"{WORK_DIR}\\buckfast_{PIPELINE_ID}_scen{SCENARIO}.out1.csv",
                     encoding = 'utf-8', skiprows = line_end - 1,
                     nrows = line_end - line_start - 3, usecols = cols)
    df_eaf = df_eaf[cols]
    df_eaf = df_eaf.iloc[1:,:].reset_index(drop = True).astype(float)

    # Rename the column names, and convert unit to kN
    cols = ['KP (m)', 'EAF Installation [RLT] (kN)', 'EAF Hydrotest (kN)',
            'EAF Operation (kN)', 'EAF Operation [without Buckling] (kN)']
    df_eaf.columns = cols
    df_eaf.loc[:, cols[1:]] = df_eaf.loc[:, cols[1:]] / 1000.0

    # Merge df_eaf and df_cbf on 'KP (m)'
    df = pd.merge(df_cbf, df_eaf, on = 'KP (m)', how = 'left')
    df = df.iloc[1:-1, :].reset_index(drop = True)

    return df

def save_buckfast_output_file(df_element, df_set_prob, df_no_buckles, df_force_prof):

    """
    Saves DataFrames to an Excel file with specified formatting.

    Parameters
    ----------
    df_element : pandas DataFrame
        DataFrame containing the element data.
    df_set_prob : pandas DataFrame
        DataFrame containing the element set data.
    df_no_buckles : pandas DataFrame
        DataFrame containing the data related to the number of buckles.
    df_force_prof : pandas DataFrame
        DataFrame containing the force profile data.

    Returns
    -------
    None
    """

    # Write the Buckfast result df into an output excel
    writer = pd.ExcelWriter(f'{WORK_DIR}\\buckfast_{PIPELINE_ID}_scen{SCENARIO}.xlsx')
    pandas.io.formats.excel.ExcelFormatter.header_style = None

    # Convert DataFrames to Excel objects
    df_element.to_excel(writer, sheet_name = 'Elements', index = False,
                        startrow = 1, header = False)
    df_set_prob.to_excel(writer, sheet_name = 'Sets',
                         index = False, startrow = 1, header = False)
    df_no_buckles.to_excel(writer, sheet_name = 'No Buckles',
                           index = False, startrow = 1, header = False)
    df_force_prof.to_excel(writer, sheet_name = 'Force Profiles',
                           index = False, startrow = 1, header = False)

    # Get the workbook and worksheet objects.
    workbook = writer.book
    worksheet1 = writer.sheets['Elements']
    worksheet2 = writer.sheets['Sets']
    worksheet3 = writer.sheets['No Buckles']
    worksheet4 = writer.sheets['Force Profiles']

    # Add generic cell formats to Excel file
    formatc1 = workbook.add_format({'num_format': '#,##0', 'align': 'center',
                                    'valign': 'vcenter', 'text_wrap': True})
    formatc2 = workbook.add_format({'num_format': '0.000', 'align': 'center',
                                    'valign': 'vcenter', 'text_wrap': True})
    formatc3 = workbook.add_format({'num_format': '#,###0.0', 'align': 'center',
                                    'valign': 'vcenter', 'text_wrap': False})
    formath1 = workbook.add_format({'num_format': '#,###', 'bold': True, 'border': 1,
                                    'bg_color': '#C0C0C0', 'align': 'center',
                                    'valign': 'vcenter', 'text_wrap': True})

    # Set the column width and format of the Excel worksheets
    worksheet1.set_column('A:B', 12.5, formatc1)
    worksheet1.set_column('C:D', 12.5, formatc2)
    worksheet1.set_column('E:H', 12.5, formatc3)

    worksheet2.set_column('A:B', 12.5, formatc1)
    worksheet2.set_column('C:D', 12.5, formatc2)
    worksheet2.set_column('E:L', 12.5, formatc3)

    worksheet3.set_column('A:B', 12.5, formatc1)
    worksheet3.set_column('C:D', 12.5, formatc2)

    worksheet4.set_column('A:A', 12.5, formatc1)
    worksheet4.set_column('B:G', 12.5, formatc3)

    # Write the colum hearders with the defined format
    for col_num, value in enumerate(df_element.columns.values):
        worksheet1.write(0, col_num, value, formath1)
    for col_num, value in enumerate(df_set_prob.columns.values):
        worksheet2.write(0, col_num, value, formath1)
    for col_num, value in enumerate(df_no_buckles.columns.values):
        worksheet3.write(0, col_num, value, formath1)
    for col_num, value in enumerate(df_force_prof.columns.values):
        worksheet4.write(0, col_num, value, formath1)

    # Close the Excel writer and output the Excel file
    writer.close()

def read_output_file(result_file_name):

    '''
    Read the data from the Buckfast or BuckPy result file.

    Parameters
    ----------
    result_file_name : string
        File path of the Buckfast or BuckPy result file.

    Returns
    -------
    df_elem : pandas DataFrame
        DataFrame containing the element data.
    df_set : pandas DataFrame
        DataFrame containing the element set data.
    df_buckle : pandas DataFrame
        DataFrame containing the data related to the number of buckles.
    df_force_prof : pandas DataFrame
        DataFrame containing the force profile data.
    df_comb_set : pandas DataFrame
        DataFrame containing the buckling combination per set data.
    df_comb_section : pandas DataFrame
        DataFrame containing the buckling combination per section data.
    '''

    # Read all tabs in the Buckfast or BuckPy result file
    all_sheets_dict = pd.read_excel(f'{WORK_DIR}\\{result_file_name}.xlsx',
                                    sheet_name = None)

    # Read 'Elements', 'Sets', 'No Buckles' and 'Force Profiles' tab
    df_elem = all_sheets_dict['Elements']
    df_sets = all_sheets_dict['Sets']
    df_buckle = all_sheets_dict['No Buckles']
    df_force_prof = all_sheets_dict['Force Profiles']

    if ('buckfast' in result_file_name) or (not PLOT_COMBINATION):
        return df_elem, df_sets, df_buckle, df_force_prof

    elif PLOT_COMBINATION:
        # Read the additional tabs of buckling combination in the BuckPy result
        df_comb_set = all_sheets_dict['Comb Buckles per Set']
        df_comb_set = df_comb_set.drop(df_comb_set.columns[0], axis = 1)
        df_comb_section = all_sheets_dict['Comb Buckles per Section']
        df_comb_section = df_comb_section.drop(df_comb_section.columns[0], axis = 1)

        # Set new header based on header and row 1, and select the first 3 most frequent combination
        cols_1 = df_comb_set.columns
        cols_2 = df_comb_set.loc[0].values.tolist()
        new_cols = np.concatenate(np.array([cols_1[0:3], cols_2[3:]], dtype = object)).tolist()
        df_comb_set.columns = new_cols
        df_comb_set = df_comb_set.iloc[1:4].reset_index(drop = True)
        df_comb_set[cols_2[3:]] = df_comb_set[cols_2[3:]].astype(int)

        cols_1 = df_comb_section.columns
        cols_2 = df_comb_section.loc[0].values.tolist()
        new_cols = np.concatenate(np.array([cols_1[0:3], cols_2[3:]], dtype = object)).tolist()
        df_comb_section.columns = new_cols
        df_comb_section = df_comb_section.iloc[1:4].reset_index(drop = True)
        df_comb_section[cols_2[3:]] = df_comb_section[cols_2[3:]].astype(int)

        return df_elem, df_sets, df_buckle, df_force_prof, df_comb_set, df_comb_section

def assembly_dataframe_plot(df_buckpy, df_buckfast):

    """
    Assemble the dataframes on probabilistic results from Buckfast and BuckPy.

    Parameters
    ----------
    df_buckpy : pandas DataFrame
        Probability of number of buckles over pipeline in the BuckPy result.
    df_buckfast : pandas DataFrame
        Probability of number of buckles over pipeline in the Buckfast result.
    """

    # Dataframe with post-processing inputs from the BuckPy input file
    all_sheets_dict = pd.read_excel(f'{WORK_DIR}\\{BUCKPY_INPUT_FILE}.xlsx', sheet_name = None)

    df_scen = all_sheets_dict['Scenario']
    df_pp = all_sheets_dict['Post-Processing']

    layout_no = df_scen.loc[(df_scen['Pipeline'] == PIPELINE_ID) &
                            (df_scen['Scenario'] == SCENARIO), 'Layout Set'].iloc[0]

    df_pp = df_pp.loc[(df_pp['Pipeline'] == PIPELINE_ID) &
                      (df_pp['Layout Set'] == layout_no)]
    df_pp = df_pp.rename(columns = {'Post-Processing Set': 'Set Label',
                                    'KP From': 'KP From (m)',
                                    'KP To': 'KP To (m)'})
    df_pp = df_pp[['Set Label', 'KP From (m)', 'KP To (m)']]

    # DataFrame with outputs from BuckPy
    df_buckpy = df_buckpy[['Set Label', 'KP From (m)', 'KP To (m)', 'Probability of Buckling',
                           'Characteristic VAS Probability',
                           'Characteristic VAS, Unconditional (m)',
                           'Characteristic Lateral Breakout Friction Probability',
                           'Characteristic Lateral Breakout Friction, Buckles',
                           'Lateral Breakout Friction, HE, Geotech']]
    df_buckpy = df_buckpy.rename(
        columns = {'Probability of Buckling': 'Probability of Buckling - BuckPy',
                   'Characteristic VAS, Unconditional (m)': 'Characteristic VAS - BuckPy'})

    # DataFrame with outputs from Buckfast
    df_buckfast = df_buckfast[['Set Label', 'Probability of Buckling',
                               'Characteristic VAS, Unconditional, Rogue (m)',
                               'Characteristic VAS, Unconditional, Planned (m)']]
    df_buckfast = df_buckfast.rename(
        columns = {'Probability of Buckling': 'Probability of Buckling - Buckfast'})

    # Merge 'KP From (m)' and 'KP To (m)' columns to Buckfast result df
    df = pd.merge(df_buckpy, df_buckfast, on = 'Set Label', how = 'outer')

    df['Characteristic VAS - Buckfast'] = df['Characteristic VAS, Unconditional, Rogue (m)']
    df.loc[df['Characteristic VAS Probability'] == 0.1, 'Characteristic VAS - Buckfast'] = \
        df['Characteristic VAS, Unconditional, Planned (m)']

    # Add sets without outputs
    df = pd.concat([df, df_pp])
    df = df.drop_duplicates(subset = ['Set Label', 'KP From (m)', 'KP To (m)'],
                            keep = 'first')
    df = df.sort_values(by = ['KP From (m)'])

    # Double each row for KP
    df = assembly_double_rows(df)

    # Add initial and final rows with zeros
    df_temp1 = pd.DataFrame({'KP': [df['KP'].min()],
                             'Probability of Buckling - BuckPy': [0.0],
                             'Probability of Buckling - Buckfasr': [0.0],
                             'Characteristic VAS - BuckPy': [0.0],
                             'Characteristic VAS - Buckfast': [0.0],
                             'Characteristic Lateral Breakout Friction, Buckles': [0.0]})
    df_temp2 = pd.DataFrame({'KP': [df['KP'].max()],
                             'Probability of Buckling - BuckPy': [0.0],
                             'Probability of Buckling - Buckfasr': [0.0],
                             'Characteristic VAS - BuckPy': [0.0],
                             'Characteristic VAS - Buckfast': [0.0],
                             'Characteristic Lateral Breakout Friction, Buckles': [0.0]})
    df = pd.concat([df_temp1, df, df_temp2], ignore_index = True)
    df = df.fillna(0.0)

    # Replace 0.0 with NaN for Geotech friction column
    df.loc[df['Characteristic Lateral Breakout Friction, Buckles'] == 0.0,
           'Lateral Breakout Friction, HE, Geotech'] = 0.0

    # Find the BuckPy friction with Prob. of Bucckling < 5% and replace it with 0.0
    df.loc[df['Probability of Buckling - BuckPy'] < 0.05,
           'Characteristic Lateral Breakout Friction, Buckles'] = 0.0

    return df

def assembly_dataframe_bend_sleeper_ilt():

    '''
    Read the route bend data from the BuckPy input file and select the KP
    of the 'Bend', 'Sleeper' and 'ILT' from the 'Route Type' column.

    Parameters
    ----------
    BUCKPY_INPUT_FILE : string
        File path of the BuckPy input file.

    Returns
    -------
    df1 : pandas DataFrame
        DataFrame containing the 'Bend' route type data.
    df2 : pandas DataFrame
        DataFrame containing the 'Sleeper' route type data.
    df3 : pandas DataFrame
        DataFrame containing the 'ILT' route type data.
    '''

    # Read all tabs in the Buckfast or BuckPy result file
    all_sheets_dict = pd.read_excel(f'{WORK_DIR}\\{BUCKPY_INPUT_FILE}.xlsx', sheet_name = None)

    # Read 'Scenario' and 'Route' tab
    df_scen = all_sheets_dict['Scenario']
    df_route = all_sheets_dict['Route']

    layout_no = df_scen.loc[(df_scen['Pipeline'] == PIPELINE_ID) &
                            (df_scen['Scenario'] == SCENARIO), 'Layout Set'].iloc[0]

    df = df_route.loc[(df_route['Pipeline'] == PIPELINE_ID) &
                      (df_route['Layout Set'] == layout_no)]

    # Select route bend and KP columns
    cols = ['Route Type', 'Point ID From', 'Point ID To', 'KP From', 'KP To']
    df = df[cols]

    # Select rows of route bend from 'Route Type'
    df1 = df.loc[df['Route Type'] == 'Bend'].copy()
    df1 = df1.reset_index(drop = True)

    # Select rows of sleeper from 'Point ID'
    df2 = df.iloc[1:-1].loc[(df['Route Type'] == 'Sleeper')].copy()
    df2 = df2.reset_index(drop = True)

    # Select rows of ILT from 'Point ID'
    df3 = df.iloc[1:-1].loc[df['Point ID From'].str.contains('ILT') &
                            df['Point ID To'].str.contains('ILT')].copy()
    df3 = df3.reset_index(drop = True)

    return df1, df2, df3

def plot_bend_sleeper_ilt(a1, df1, df2, df3, y_max):

    """
    Plot route bend, sleeper and ILT.

    Parameters
    ----------
    a1 : Matplotlib plot
        Axis of the Matplotlib plot.
    df1 : pandas DataFrame
        Dataframe containing the locations of the route bends.
    df2 : pandas DataFrame
        Dataframe containing the locations of sleepers.
    df3 : pandas DataFrame
        Dataframe containing the locations of ILTs.
    """

    # Plot bends
    for index, row in df1.iterrows():
        if index == 0:
            a1.plot([row['KP From'], row['KP To']], 2 * [0.1 * y_max],
                    color = 'C2', label = 'Route Bend', linestyle = 'dotted')
        else:
            a1.plot([row['KP From'], row['KP To']], 2 * [0.1 * y_max],
                    color = 'C2', linestyle = 'dotted')

    # Plot sleepers
    for index, row in df2.iterrows():
        kp_centre = 0.5 * (row['KP From'] + row['KP To'])
        if index == 0:
            a1.scatter(kp_centre, 0.1 * y_max,
                       color = 'C3', label = 'Sleeper', marker = '*')
        else:
            a1.scatter(kp_centre, 0.1 * y_max,
                       color = 'C3', marker = '*')

    # Plot ILTs
    for index, row in df3.iterrows():
        kp_centre = 0.5 * (row['KP From'] + row['KP To'])
        if index == 0:
            a1.scatter(kp_centre, 0.1 * y_max,
                       color = 'C4', label = 'ILT', marker = '+')
        else:
            a1.scatter(kp_centre, 0.1 * y_max,
                       color = 'C4', marker = '+')

def plot_force_profiles(df_dict, column_name, file_string):

    """
    Plot probabilistic results of the unbuckled and buckled EAF from Buckfast and BuckPy.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.
    column_name : String
        The EAF column name in the dataframe.
    file_string : String
        'unbuckled' or 'buckled' conditions.
    """

    if PLOT_CURRENT_SCENARIO:

        for sce_no in SCENARIO_LIST:

            # Obtain dataframe for current scenario in the dictionary
            df_buckpy = df_dict[f"df_force_buckpy_{sce_no}"]
            df_buckfast = df_dict[f"df_force_buckfast_{sce_no}"]

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot the unbuckled and buckled EAF
            a1 = fig.add_subplot(111)
            a1.plot(df_buckpy['KP (m)'].sort_values().reset_index(drop = True),
                    df_buckpy[column_name], label = 'BuckPy')
            a1.plot(df_buckfast['KP (m)'].sort_values().reset_index(drop = True),
                    df_buckfast[column_name], label = 'Buckfast', linestyle = 'dashed')

            a1.set_xlabel('KP (m)')
            a1.set_ylabel('EAF (kN)')
            a1.legend()
            a1.grid()

            plt.savefig(
                f"{WORK_DIR}\\zplot_{PIPELINE_ID}_force_{file_string}_scen[{sce_no}].tiff",
                dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

    if PLOT_ALL_SCENARIOS:

        for software in ['buckfast', 'buckpy']:

            # Define The name string in output plot, 'Buckfast' or 'BuckPy'
            tiff_str = 'Buckfast' if software == 'buckfast' else 'BuckPy'

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot the unbuckled and buckled EAF
            a1 = fig.add_subplot(111)

            for sce_no in SCENARIO_LIST:
                df = df_dict[f"df_force_{software}_{sce_no}"]
                a1.plot(df['KP (m)'].sort_values().reset_index(drop = True), df[column_name],
                        label = f'Scenario {sce_no}', linestyle = 'dashed')

            a1.set_xlabel('KP (m)')
            a1.set_ylabel('EAF (kN)')
            a1.legend()
            a1.grid()

            plt.savefig(
                f"{WORK_DIR}\\zplot_{PIPELINE_ID}_force_{file_string}_{tiff_str}_scen{SCENARIO_LIST}.tiff",
                dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

def plot_no_buckles(df_dict):

    """
    Plot probabilistic results of the number of buckles from Buckfast and BuckPy.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.
    """

    if PLOT_CURRENT_SCENARIO:

        for sce_no in SCENARIO_LIST:

            # Obtain dataframe for current scenario in the dictionary
            cols = ['Number of Buckles', 'Probability of Buckling']
            df_buckpy = df_dict[f"df_buckle_buckpy_{sce_no}"][cols]
            df_buckfast = df_dict[f"df_buckle_buckfast_{sce_no}"][cols]

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot distribution of number of buckles
            a1 = fig.add_subplot(111)

            # Add dataframes into a dictionary
            dfs = {}
            dfs["df_buckpy"] = df_buckpy
            dfs["df_buckfast"] = df_buckfast

            # Obtain the common No. of buckles for all the dataframes
            total_no_buckles = []
            for count, df in enumerate(dfs):
                total_no_buckles = pd.Series(sorted(set(
                    np.concatenate((total_no_buckles, dfs[df]['Number of Buckles'])))))
            df_temp = pd.DataFrame({'Number of Buckles': total_no_buckles})

            # Set the width and the positions of the bars
            bar_width = 0.35
            no_bar = len(dfs)

            label_list = ['BuckPy', 'Buckfast']
            for count, df in enumerate(dfs):

                # Merge the No. of buckles to each dataframe and fill the additional prob. with 0s
                dfs[df] = pd.merge(df_temp, dfs[df], on = 'Number of Buckles', how = 'left')
                dfs[df] = dfs[df].fillna(0.0)

                # Create bar plots
                x_labels = dfs[df]['Number of Buckles'].astype(int)
                x_values = x_labels + count * bar_width
                a1.bar(x_values, dfs[df]['Probability of Buckling'], width = bar_width,
                       alpha = 0.75, label = label_list[count])
            x_ticks = x_labels + bar_width / no_bar

            a1.set_xlabel('Number of Buckles')
            a1.set_ylabel('Probability of Buckling')
            a1.legend()
            a1.grid()
            a1.set_xticks(x_ticks.tolist())
            a1.set_xticklabels(x_labels.tolist())

            plt.savefig(f"{WORK_DIR}\\zplot_{PIPELINE_ID}_no_buckles_scen[{sce_no}].tiff",
                        dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

    if PLOT_ALL_SCENARIOS:

        for software in ['buckfast', 'buckpy']:

            # Define The name string in output plot, 'Buckfast' or 'BuckPy'
            tiff_str = 'Buckfast' if software == 'buckfast' else 'BuckPy'

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot distribution of number of buckles
            a1 = fig.add_subplot(111)

            # Obtain the common No. of buckles for all the dataframes
            total_no_buckles = []
            for sce_no in SCENARIO_LIST:
                df = df_dict[f"df_buckle_{software}_{sce_no}"]
                total_no_buckles = pd.Series(sorted(set(
                    np.concatenate((total_no_buckles, df['Number of Buckles'])))))
            df_temp = pd.DataFrame({'Number of Buckles': total_no_buckles})

            # Set the width and the positions of the bars
            no_bar = len(SCENARIO_LIST)
            bar_width = 0.95 / no_bar

            for count, sce_no in enumerate(SCENARIO_LIST):

                # Merge the No. of buckles to each dataframe and fill the additional prob. with 0s
                df = df_dict[f"df_buckle_{software}_{sce_no}"]
                df = pd.merge(df_temp, df, on = 'Number of Buckles', how = 'left')
                df = df.fillna(0.0)

                # Create bar plots
                x_labels = df['Number of Buckles'].astype(int)
                x_values = x_labels + count * bar_width
                a1.bar(x_values, df['Probability of Buckling'], width = bar_width,
                       alpha = 0.75, label = f'Scenario {sce_no}')
            x_ticks = x_labels + bar_width / 2 * (no_bar - 1.0)

            a1.set_xlabel('Number of Buckles')
            a1.set_ylabel('Probability of Buckling')
            a1.legend()
            a1.grid()
            a1.set_xticks(x_ticks.tolist())
            a1.set_xticklabels(x_labels.tolist())

            plt.savefig(
                f"{WORK_DIR}\\zplot_{PIPELINE_ID}_no_buckles_{tiff_str}_scen{SCENARIO_LIST}.tiff",
                dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

def plot_prob_buckling(df_dict):

    """
    Plot probabilities of buckling per kilometre.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.
    """

    if PLOT_CURRENT_SCENARIO:

        for sce_no in SCENARIO_LIST:

            # Obtain dataframe for current scenario in the dictionary
            df = df_dict[f"df_set_{sce_no}"]
            df1 = df_dict[f"df_bend_{sce_no}"]
            df2 = df_dict[f"df_sleeper_{sce_no}"]
            df3 = df_dict[f"df_ilt_{sce_no}"]

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot probabilities of buckling
            a1 = fig.add_subplot(111)
            a1.plot(df['KP'], df['Probability of Buckling - BuckPy'],
                    label = 'BuckPy')
            a1.plot(df['KP'], df['Probability of Buckling - Buckfast'],
                    label = 'Buckfast', linestyle = 'dashed')

            # Plot route bend, sleeper and ILT
            plot_bend_sleeper_ilt(a1, df1, df2, df3, df['Probability of Buckling - BuckPy'].max())

            a1.set_xlabel('KP (m)')
            a1.set_ylabel('Probability of Buckling')
            a1.legend()
            a1.grid()

            plt.savefig(f'{WORK_DIR}\\zplot_{PIPELINE_ID}_prob_buckling_scen[{sce_no}].tiff',
                        dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

    if PLOT_ALL_SCENARIOS:

        for software in ['buckfast', 'buckpy']:

            # Define The name string in output plot, 'Buckfast' or 'BuckPy'
            tiff_str = 'Buckfast' if software == 'buckfast' else 'BuckPy'

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot probabilities of buckling
            a1 = fig.add_subplot(111)

            for sce_no in SCENARIO_LIST:
                df = df_dict[f"df_set_{sce_no}"]
                a1.plot(df['KP'], df[f'Probability of Buckling - {tiff_str}'],
                        label = f'Scenario {sce_no}', linestyle = 'dashed')

            # Plot route bend, sleeper and ILT
            df1 = df_dict[f"df_bend_{sce_no}"]
            df2 = df_dict[f"df_sleeper_{sce_no}"]
            df3 = df_dict[f"df_ilt_{sce_no}"]
            plot_bend_sleeper_ilt(a1, df1, df2, df3, df['Probability of Buckling - BuckPy'].max())

            a1.set_xlabel('KP (m)')
            a1.set_ylabel('Probability of Buckling')
            a1.legend()
            a1.grid()

            plt.savefig(
                f'{WORK_DIR}\\zplot_{PIPELINE_ID}_prob_buckling_{tiff_str}_scen{SCENARIO_LIST}.tiff',
                dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

def plot_vas(df_dict):

    """
    Plot characteristic VAS per kilometre.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.
    """

    if PLOT_CURRENT_SCENARIO:

        for sce_no in SCENARIO_LIST:

            # Obtain dataframe for current scenario in the dictionary
            df = df_dict[f"df_set_{sce_no}"]
            df1 = df_dict[f"df_bend_{sce_no}"]
            df2 = df_dict[f"df_sleeper_{sce_no}"]
            df3 = df_dict[f"df_ilt_{sce_no}"]

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot characteristic VAS
            a1 = fig.add_subplot(111)
            a1.plot(df['KP'], df['Characteristic VAS - BuckPy'],
                    label = 'BuckPy')
            a1.plot(df['KP'], df['Characteristic VAS - Buckfast'],
                    label = 'Buckfast', linestyle = 'dashed')

            # Plot route bend, sleeper and ILT
            plot_bend_sleeper_ilt(a1, df1, df2, df3, df['Characteristic VAS - BuckPy'].max())

            a1.set_xlabel('KP (m)')
            a1.set_ylabel('Characteristic VAS (m)')
            a1.legend()
            a1.grid()

            plt.savefig(f'{WORK_DIR}\\zplot_{PIPELINE_ID}_vas_scen[{sce_no}].tiff',
                        dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

    if PLOT_ALL_SCENARIOS:

        for software in ['buckfast', 'buckpy']:

            # Define The name string in output plot, 'Buckfast' or 'BuckPy'
            tiff_str = 'Buckfast' if software == 'buckfast' else 'BuckPy'

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot characteristic VAS
            a1 = fig.add_subplot(111)

            for sce_no in SCENARIO_LIST:
                df = df_dict[f"df_set_{sce_no}"]
                a1.plot(df['KP'], df[f'Characteristic VAS - {tiff_str}'],
                        label = f'Scenario {sce_no}', linestyle = 'dashed')

            # Plot route bend, sleeper and ILT
            df1 = df_dict[f"df_bend_{sce_no}"]
            df2 = df_dict[f"df_sleeper_{sce_no}"]
            df3 = df_dict[f"df_ilt_{sce_no}"]
            plot_bend_sleeper_ilt(a1, df1, df2, df3, df['Characteristic VAS - BuckPy'].max())

            a1.set_xlabel('KP (m)')
            a1.set_ylabel('Characteristic VAS (m)')
            a1.legend()
            a1.grid()

            plt.savefig(
                f'{WORK_DIR}\\zplot_{PIPELINE_ID}_vas_{tiff_str}_scen{SCENARIO_LIST}.tiff',
                dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

def plot_friction(df_dict):

    """
    Plot characteristic friction per kilometre.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.
    """

    if PLOT_CURRENT_SCENARIO:

        for sce_no in SCENARIO_LIST:

            # Obtain dataframe for current scenario in the dictionary
            df = df_dict[f"df_set_{sce_no}"].copy()
            df1 = df_dict[f"df_bend_{sce_no}"]
            df2 = df_dict[f"df_sleeper_{sce_no}"]
            df3 = df_dict[f"df_ilt_{sce_no}"]

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot characteristic friction
            a1 = fig.add_subplot(111)
            a1.plot(df['KP'], df['Characteristic Lateral Breakout Friction, Buckles'],
                    label = 'Characteristic Friction, Buckles')
            a1.plot(df['KP'], df['Lateral Breakout Friction, HE, Geotech'],
                    label = 'Geotechnical Friction, HE', linestyle = 'dashed')

            # Plot route bend, sleeper and ILT
            plot_bend_sleeper_ilt(a1, df1, df2, df3,
                                  df['Characteristic Lateral Breakout Friction, Buckles'].max())

            a1.set_xlabel('KP (m)')
            a1.set_ylabel('Characteristic Friction')
            a1.legend()
            a1.grid()

            plt.savefig(f'{WORK_DIR}\\zplot_{PIPELINE_ID}_friction_scen[{sce_no}].tiff',
                        dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

    if PLOT_ALL_SCENARIOS:

        # Define The name string in output plot, 'BuckPy'
        tiff_str = 'BuckPy'

        # Generate matplolib figure
        fig = plt.figure()
        dpi_size = 110
        fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

        # Plot characteristic friction
        a1 = fig.add_subplot(111)

        for sce_no in SCENARIO_LIST:
            df = df_dict[f"df_set_{sce_no}"].copy()

            a1.plot(df['KP'], df['Characteristic Lateral Breakout Friction, Buckles'],
                    label = f'Scenario {sce_no}',
                    linestyle = 'dashed')

        # Plot route bend, sleeper and ILT
        df1 = df_dict[f"df_bend_{sce_no}"]
        df2 = df_dict[f"df_sleeper_{sce_no}"]
        df3 = df_dict[f"df_ilt_{sce_no}"]
        plot_bend_sleeper_ilt(a1, df1, df2, df3,
                                df['Characteristic Lateral Breakout Friction, Buckles'].max())

        a1.set_xlabel('KP (m)')
        a1.set_ylabel('Characteristic Friction')
        a1.legend()
        a1.grid()

        plt.savefig(
            f'{WORK_DIR}\\zplot_{PIPELINE_ID}_friction_{tiff_str}_scen{SCENARIO_LIST}.tiff',
            dpi = dpi_size)
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()

def assembly_dataframe_scenarios(*args):

    """
    Assemble the dataframes of buckfast and buckpy results of all scenarios.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.
    df_sets_buckpy : pandas DataFrame
        DataFrame containing the element set data of BuckPy.
    df_buckles_buckpy : pandas DataFrame
        DataFrame containing the data related to the number of buckles of BuckPy.
    df_forces_buckpy : pandas DataFrame
        DataFrame containing the force profile data of BuckPy.
    df_sets_buckfast : pandas DataFrame
        DataFrame containing the element set data of Buckfast.
    df_buckles_buckfast : pandas DataFrame
        DataFrame containing the data related to the number of buckles of Buckfast.
    df_forces_buckfast : pandas DataFrame
        DataFrame containing the force profile data of Buckfast.
    df_sets : pandas DataFrame
        DataFrame containing the element set data of BuckPy and Buckfast.
    df_bends : pandas DataFrame
        DataFrame containing the 'Bend' route type data.
    df_sleepers : pandas DataFrame
        DataFrame containing the 'Sleeper' route type data.
    df_ilts : pandas DataFrame
        DataFrame containing the 'ILT' route type data.
    df_comb_set : pandas DataFrame
        DataFrame containing the combination of buckles per set data of BuckPy.
    df_comb_section : pandas DataFrame
        DataFrame containing the combination of buckles per section data of BuckPy.

    Returns
    -------
    df_dict_sce : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.
    """

    if PLOT_COMBINATION:
        df_dict, df_sets_buckpy, df_buckles_buckpy, df_forces_buckpy, df_sets_buckfast,\
            df_buckles_buckfast, df_forces_buckfast, df_sets, df_bends, df_sleepers, df_ilts,\
                df_comb_set, df_comb_section = args
    else:
        df_dict, df_sets_buckpy, df_buckles_buckpy, df_forces_buckpy, df_sets_buckfast,\
            df_buckles_buckfast, df_forces_buckfast, df_sets, df_bends, df_sleepers, df_ilts = args

    df_dict[f"df_set_buckpy_{SCENARIO}"] = df_sets_buckpy
    df_dict[f"df_buckle_buckpy_{SCENARIO}"] = df_buckles_buckpy
    df_dict[f"df_force_buckpy_{SCENARIO}"] = df_forces_buckpy
    df_dict[f"df_set_buckfast_{SCENARIO}"] = df_sets_buckfast
    df_dict[f"df_buckle_buckfast_{SCENARIO}"] = df_buckles_buckfast
    df_dict[f"df_force_buckfast_{SCENARIO}"] = df_forces_buckfast
    df_dict[f"df_set_{SCENARIO}"] = df_sets
    df_dict[f"df_bend_{SCENARIO}"] = df_bends
    df_dict[f"df_sleeper_{SCENARIO}"] = df_sleepers
    df_dict[f"df_ilt_{SCENARIO}"] = df_ilts

    if PLOT_COMBINATION:
        df_dict[f"df_comb_set_{SCENARIO}"] = df_comb_set
        df_dict[f"df_comb_section_{SCENARIO}"] = df_comb_section

    return df_dict

def assembly_dataframe_summary(df_dict):

    """
    Create the dataframes of buckling and characteristic VAS results for summary table.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.

    Returns
    -------
    df_buckle : pandas Dataframe
        Dataframe containing the buckling results of Buckfast and BuckPy
        in all scenarios.
    df_vas : pandas Dataframe
        Dataframe containing the characteristic VAS results of Buckfast and BuckPy
        in all scenarios.
    """

    # Create empty dataframes for buckling and characteristic VAS summary table
    df_buckle = pd.DataFrame()
    df_vas = pd.DataFrame()

    for index, sce_no in enumerate(SCENARIO_LIST):

        # Obtain the result dataframe of the current scenario
        df_buckles_buckfast = df_dict[f"df_buckle_buckfast_{sce_no}"]
        df_buckles_buckpy = df_dict[f"df_buckle_buckpy_{sce_no}"]

        # Create the dataframe for tab 'Buckling' of the current scenario
        df_buckle.loc[index, 'Scenario'] = sce_no
        df_buckle.loc[index, 'Buckfast - Probability of Buckling'] = \
            1.0 - df_buckles_buckfast.loc[0, 'Probability of Buckling']
        df_buckle.loc[index, 'Buckfast - Number of Buckles - Min'] = \
            df_buckles_buckfast['Number of Buckles'].min()
        df_buckle.loc[index, 'Buckfast - Number of Buckles - Mode'] = df_buckles_buckfast.loc[
            df_buckles_buckfast['Probability of Buckling'].idxmax(), 'Number of Buckles']
        df_buckle.loc[index, 'Buckfast - Number of Buckles - Max'] = \
            df_buckles_buckfast['Number of Buckles'].max()
        df_buckle.loc[index, 'BuckPy - Probability of Buckling'] = \
            1.0 - df_buckles_buckpy.loc[0, 'Probability of Buckling']
        df_buckle.loc[index, 'BuckPy - Number of Buckles - Min'] = \
            df_buckles_buckpy['Number of Buckles'].min()
        df_buckle.loc[index, 'BuckPy - Number of Buckles - Mode'] = df_buckles_buckpy.loc[
            df_buckles_buckpy['Probability of Buckling'].idxmax(), 'Number of Buckles']
        df_buckle.loc[index, 'BuckPy - Number of Buckles - Max'] = \
            df_buckles_buckpy['Number of Buckles'].max()

        # Obtain the result dataframe of the current scenario
        df_vas_buckfast = df_dict[f"df_set_buckfast_{sce_no}"]
        df_vas_buckpy = df_dict[f"df_set_buckpy_{sce_no}"]

        # Create the dataframe for tab 'Characteristic' of the current scenario
        cols_buckfast = ['Set Label', 'Probability of Buckling',
                         'Characteristic VAS, Unconditional, Rogue (m)']
        cols_buckpy = ['Set Label', 'KP From (m)', 'KP To (m)', 'Probability of Buckling',
                       'Characteristic VAS, Unconditional (m)',
                       'Characteristic Lateral Breakout Friction, Buckles']
        df_merged = pd.merge(df_vas_buckpy[cols_buckpy], df_vas_buckfast[cols_buckfast],
                             on = 'Set Label', how = 'left')

        # Rename and reorder columns
        df_merged.columns = [
            'Element Set - Identifier', 'Element Set - KP From (m)', 'Element - KP To (m)',
            'BuckPy - Probability of Buckling', 'BuckPy - Characteristic VAS (m)',
            'BuckPy - Characteristic Peak Lateral Friction', 'Buckfast - Probability of Buckling',
            'Buckfast - Characteristic VAS (m)']
        new_cols = ['Element Set - Identifier', 'Element Set - KP From (m)', 'Element - KP To (m)',
                    'Buckfast - Probability of Buckling', 'Buckfast - Characteristic VAS (m)',
                    'BuckPy - Probability of Buckling', 'BuckPy - Characteristic VAS (m)',
                    'BuckPy - Characteristic Peak Lateral Friction']
        df_merged = df_merged[new_cols]

        # Calculate the maxinum value of characteristic VAS and friction
        df_merged['Buckfast - Max Char. VAS (m)'] = \
            df_merged['Buckfast - Characteristic VAS (m)'].max()
        df_merged['BuckPy - Max Char. VAS (m)'] = \
            df_merged['BuckPy - Characteristic VAS (m)'].max()
        df_merged['BuckPy - Max Char. Friction'] = \
            df_merged['BuckPy - Characteristic Peak Lateral Friction'].max()

        # Insert pipeline and scenario columns to the first and second column
        df_merged.insert(0, 'Pipeline', PIPELINE_ID)
        df_merged.insert(1, 'Scenario', sce_no)
        df_vas = pd.concat([df_vas, df_merged], ignore_index = True)

    # Insert pipeline column to the first column
    df_buckle.insert(0, 'Pipeline', PIPELINE_ID)

    # Read Scenario Description from BuckPy input file
    df_sce = pd.read_excel(f"{WORK_DIR}\\{BUCKPY_INPUT_FILE}.xlsx", sheet_name = 'Scenario')
    df_sce = df_sce[['Pipeline', 'Scenario', 'Scenario Description']]

    # Merge description to the dataframe of the two tabs on 'Pipeline' and 'Scenario'
    df_buckle = pd.merge(df_buckle, df_sce, on = ['Pipeline', 'Scenario'], how = 'left')
    df_vas = pd.merge(df_vas, df_sce, on = ['Pipeline', 'Scenario'], how = 'left')

    # Insert the description column to the third column
    col_buckle = df_buckle.pop('Scenario Description')
    df_buckle.insert(2, 'Scenario Description', col_buckle)
    col_vas = df_vas.pop('Scenario Description')
    df_vas.insert(2, 'Scenario Description', col_vas)

    return df_buckle, df_vas

def assembly_dataframe_concat_summary(df_dict):

    """
    Concatenate the dataframes of buckling and characteristic VAS results of different
    pipelines for summary table.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all pipelines.

    Returns
    -------
    df_buckle : pandas Dataframe
        Dataframe containing the buckling results of Buckfast and BuckPy
        in all pipelines.
    df_vas : pandas Dataframe
        Dataframe containing the characteristic VAS results of Buckfast and BuckPy
        in all pipelines.
    """

    # Create empty dataframes for buckling and characteristic VAS summary table
    df_buckle = pd.DataFrame()
    df_vas = pd.DataFrame()

    for index, pipe in enumerate(PIPELINE_LIST):

        # Obtain the result dataframe of the current pipeline
        df_buckle_temp = df_dict[f"df_buckling_{pipe}"]
        df_vas_temp = df_dict[f"df_char_vas_{pipe}"]

        # Concatenate the results for each pipeline
        df_buckle = pd.concat([df_buckle, df_buckle_temp], ignore_index = True)
        df_vas = pd.concat([df_vas, df_vas_temp], ignore_index = True)

    return df_buckle, df_vas

def save_summary_table_file(df_buckle, df_vas):

    """
    Saves DataFrames to an Excel file with specified formatting.

    Parameters
    ----------
    df_buckle : pandas Dataframe
        Dataframe containing the buckling results of Buckfast and BuckPy
        in all pipelines.
    df_vas : pandas Dataframe
        Dataframe containing the characteristic VAS results of Buckfast and BuckPy
        in all pipelines.
    """

    # Write the result df into an output excel
    writer = pd.ExcelWriter(f'{WORK_DIR}\\a_summary_table.xlsx')
    pandas.io.formats.excel.ExcelFormatter.header_style = None

    # Convert DataFrames to Excel objects
    df_buckle.to_excel(writer, sheet_name = 'Buckling', index = False,
                       startrow = 1, header = False)
    df_vas.to_excel(writer, sheet_name = 'Characteristic',
                    index = False, startrow = 1, header = False)

    # Get the workbook and worksheet objects.
    workbook = writer.book
    worksheet1 = writer.sheets['Buckling']
    worksheet2 = writer.sheets['Characteristic']

    # Add generic cell formats to Excel file
    formatc1 = workbook.add_format({'num_format': '#,##0', 'align': 'center',
                                    'valign': 'vcenter', 'text_wrap': True})
    formatc2 = workbook.add_format({'num_format': '0.000', 'align': 'center',
                                    'valign': 'vcenter', 'text_wrap': True})
    formatc3 = workbook.add_format({'num_format': '0%', 'align': 'center',
                                    'valign': 'vcenter', 'text_wrap': False})
    formatc4 = workbook.add_format({'num_format': '#,##0', 'align': 'left',
                                    'valign': 'vcenter', 'text_wrap': True})
    formath1 = workbook.add_format({'num_format': '#,###', 'bold': True, 'border': 1,
                                    'bg_color': '#C0C0C0', 'align': 'center',
                                    'valign': 'vcenter', 'text_wrap': True})

    # Set the column width and format of the Excel worksheets
    worksheet1.set_column('A:B', 12.5, formatc1)
    worksheet1.set_column('C:C', 83.0, formatc4)
    worksheet1.set_column('D:D', 12.5, formatc3)
    worksheet1.set_column('E:G', 12.5, formatc1)
    worksheet1.set_column('H:H', 12.5, formatc3)
    worksheet1.set_column('I:K', 12.5, formatc1)

    worksheet2.set_column('A:B', 12.5, formatc1)
    worksheet2.set_column('C:C', 83.0, formatc4)
    worksheet2.set_column('D:F', 12.5, formatc1)
    worksheet2.set_column('G:G', 12.5, formatc3)
    worksheet2.set_column('H:H', 12.5, formatc1)
    worksheet2.set_column('I:I', 12.5, formatc3)
    worksheet2.set_column('J:J', 12.5, formatc1)
    worksheet2.set_column('K:K', 12.5, formatc2)
    worksheet2.set_column('L:M', 12.5, formatc1)
    worksheet2.set_column('N:N', 12.5, formatc2)

    # Write the colum hearders with the defined format
    for col_num, value in enumerate(df_buckle.columns.values):
        worksheet1.write(0, col_num, value, formath1)
    for col_num, value in enumerate(df_vas.columns.values):
        worksheet2.write(0, col_num, value, formath1)

    # Close the Excel writer and output the Excel file
    writer.close()

def assembly_double_rows(df):

    """
    Double each row of the dataframe based on KP.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the KP and results.

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing the doubled rows of KP and results.
    """

    # Double each row of the df
    df_temp1 = df.copy()
    df_temp1 = df_temp1.rename(columns = {'KP To (m)': 'KP'})
    df_temp1 = df_temp1.drop(labels = 'KP From (m)', axis = 1)
    df_temp1['sort'] = 1

    df_temp2 = df.copy()
    df_temp2 = df_temp2.rename(columns = {'KP From (m)': 'KP'})
    df_temp2 = df_temp2.drop(labels = 'KP To (m)', axis = 1)
    df_temp2['sort'] = 2

    # Double the rows except the start and end
    df = pd.concat([df_temp1, df_temp2])
    df = df.sort_values(by = ['KP', 'sort']).reset_index(drop = True)

    # Double the start and end rows and fill nan with 0
    first_row = pd.DataFrame({'KP': [df['KP'].min()]})
    last_row = pd.DataFrame({'KP': [df['KP'].max()]})
    df = pd.concat([first_row, df, last_row], ignore_index = True)
    df = df.fillna(0)

    return df

def assembly_dataframe_combination_plot(df_comb_set, df_comb_section):

    """
    Assemble the dataframes on probabilistic results of the number of buckles from BuckPy.

    Parameters
    ----------
    df_comb_set : pandas DataFrame
        DataFrame containing the buckling combination per set data.
    df_comb_section : pandas DataFrame
        DataFrame containing the buckling combination per section data.
    
    Returns
    -------
    df_comb_set_out : pandas DataFrame
        DataFrame containing the buckling combination per set data.
    df_comb_section_out : pandas DataFrame
        DataFrame containing the buckling combination per section data.
    """

    # Dataframe with post-processing inputs from the BuckPy input file
    all_sheets_dict = pd.read_excel(f'{WORK_DIR}\\{BUCKPY_INPUT_FILE}.xlsx', sheet_name = None)

    df_scen = all_sheets_dict['Scenario']
    df_route = all_sheets_dict['Route']
    df_pp = all_sheets_dict['Post-Processing']

    layout_no = df_scen.loc[(df_scen['Pipeline'] == PIPELINE_ID) &
                            (df_scen['Scenario'] == SCENARIO), 'Layout Set'].iloc[0]

    # Select the route data of current scenario
    df_route = df_route.loc[(df_route['Pipeline'] == PIPELINE_ID) &
                            (df_route['Layout Set'] == layout_no)]
    df_route = df_route[['Point ID From', 'Point ID To', 'KP From', 'KP To']
                        ].iloc[1:-1].reset_index(drop = True)
    df_route = df_route.reset_index().rename(columns = {
        'index': 'Set Label','KP From': 'KP From (m)', 'KP To': 'KP To (m)'})
    df_route['Set Label'] = df_route['Set Label'] + 1

    # Select the post-processing data of current scenario
    df_pp = df_pp.loc[(df_pp['Pipeline'] == PIPELINE_ID) &
                      (df_pp['Layout Set'] == layout_no)]
    df_pp = df_pp.rename(columns = {'KP From': 'KP From (m)', 'KP To': 'KP To (m)'})
    df_pp = df_pp.reset_index(drop = True).reset_index().rename(columns = {'index': 'Set Label'})
    df_pp['Set Label'] = df_pp['Set Label'] + 1
    df_pp = df_pp[['Set Label', 'KP From (m)', 'KP To (m)']]

    # Transpose buckling combination dataframes
    df_comb_set_out = df_comb_set.iloc[:, 3:].T.reset_index()

    # Rename columns and add probability of buckling columns for the most frequent combinations
    df_comb_set_out.columns.values[0] = 'Ranges'
    for item in range(df_comb_set_out.shape[1] - 1):
        df_comb_set_out.columns.values[item + 1] = COL_LIST[item]
        df_comb_set_out[f"Prob. - {COL_LIST[item]}"] = \
            df_comb_set.loc[item, 'Probability of Combination']

    # Add 'Set Label' column
    df_comb_set_out = df_comb_set_out.reset_index().rename(columns = {'index': 'Set Label'})
    df_comb_set_out['Set Label'] = df_comb_set_out['Set Label'] + 1

    # Transpose buckling combination dataframes
    df_comb_section_out = df_comb_section.iloc[:, 3:].T.reset_index()

    # Rename columns and add probability of buckling columns for the most frequent combinations
    df_comb_section_out.columns.values[0] = 'Ranges'
    for item in range(df_comb_section_out.shape[1] - 1):
        df_comb_section_out.columns.values[item + 1] = COL_LIST[item]
        df_comb_section_out[f"Prob. - {COL_LIST[item]}"] = \
            df_comb_section.loc[item, 'Probability of Combination']

    # Add 'Set Label' column
    df_comb_section_out = df_comb_section_out.reset_index().rename(columns = {'index': 'Set Label'})
    df_comb_section_out['Set Label'] = df_comb_section_out['Set Label'] + 1

    # Add KP columns and double each row
    df_comb_set_out = pd.merge(df_pp, df_comb_set_out, on = 'Set Label', how = 'left')
    df_comb_set_out = assembly_double_rows(df_comb_set_out)
    df_comb_section_out = pd.merge(df_route, df_comb_section_out, on = 'Set Label', how = 'left')
    df_comb_section_out = assembly_double_rows(df_comb_section_out)

    return df_comb_set_out, df_comb_section_out

def plot_comb_per_set(df_dict):

    """
    Plot the number of buckles per kilometer for the first 3 most frequent combination per set.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.
    """

    if PLOT_CURRENT_SCENARIO:

        for sce_no in SCENARIO_LIST:

            # Obtain dataframe for current scenario in the dictionary
            df = df_dict[f"df_comb_set_{sce_no}"]
            df1 = df_dict[f"df_bend_{sce_no}"]
            df2 = df_dict[f"df_sleeper_{sce_no}"]
            df3 = df_dict[f"df_ilt_{sce_no}"]

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot distribution of number of buckles
            a1 = fig.add_subplot(111)

            # Plot the probability of the buckling combination
            # for item in range(df.filter(like = 'Prob').shape[1]):
            for item in range(1):
                prob = df.loc[1, f'Prob. - {COL_LIST[item]}'] * 100.0
                a1.plot(df['KP'], df[COL_LIST[item]], linestyle = LINESTYLE_LIST[item],
                        label = f"{COL_LIST[item]} Comb. [Prob. {prob:.1f}%]")

            # Plot route bend, sleeper and ILT
            plot_bend_sleeper_ilt(a1, df1, df2, df3, df['1st Freq.'].max())

            a1.set_xlabel('KP (m)')
            a1.set_ylabel('Number of Buckles')
            a1.legend()
            a1.grid()

            plt.savefig(f'{WORK_DIR}\\zplot_{PIPELINE_ID}_comb_per_set_scen[{sce_no}].tiff',
                        dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

    if PLOT_ALL_SCENARIOS:

        # Generate matplolib figure
        fig = plt.figure()
        dpi_size = 110
        fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

        # Plot distribution of number of buckles
        a1 = fig.add_subplot(111)
        for sce_no in SCENARIO_LIST:
            df = df_dict[f"df_comb_set_{sce_no}"]

            # Plot the probability of the buckling combination
            # for item in range(df.filter(like = 'Prob').shape[1]):
            for item in range(1):
                prob = df.loc[1, f'Prob. - {COL_LIST[item]}'] * 100.0
                a1.plot(df['KP'], df[COL_LIST[item]], linestyle = LINESTYLE_LIST[item],
                        label = f"Scenario {sce_no} {COL_LIST[item]} Comb. [Prob. {prob:.1f}%]")

        # Plot route bend, sleeper and ILT
        df1 = df_dict[f"df_bend_{sce_no}"]
        df2 = df_dict[f"df_sleeper_{sce_no}"]
        df3 = df_dict[f"df_ilt_{sce_no}"]
        plot_bend_sleeper_ilt(a1, df1, df2, df3, df['1st Freq.'].max())

        a1.set_xlabel('KP (m)')
        a1.set_ylabel('Number of Buckles')
        a1.legend()
        a1.grid()

        plt.savefig(
            f'{WORK_DIR}\\zplot_{PIPELINE_ID}_comb_per_set_BuckPy_scen{SCENARIO_LIST}.tiff',
            dpi = dpi_size)
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()

def plot_comb_per_section(df_dict):

    """
    Plot the number of buckles per kilometer for the first 3 most frequent combination per section.

    Parameters
    ----------
    df_dict : Dictionary
        Dictionary containing the Dataframes of Buckfast and BuckPy in all scenarios.
    """

    if PLOT_CURRENT_SCENARIO:

        for sce_no in SCENARIO_LIST:

            # Obtain dataframe for current scenario in the dictionary
            df = df_dict[f"df_comb_section_{sce_no}"]
            df1 = df_dict[f"df_bend_{sce_no}"]
            df2 = df_dict[f"df_sleeper_{sce_no}"]
            df3 = df_dict[f"df_ilt_{sce_no}"]

            # Generate matplolib figure
            fig = plt.figure()
            dpi_size = 110
            fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

            # Plot distribution of number of buckles
            a1 = fig.add_subplot(111)

            # Plot the probability of the buckling combination
            # for item in range(df.filter(like = 'Prob').shape[1]):
            for item in range(1):
                prob = df.loc[1, f'Prob. - {COL_LIST[item]}'] * 100.0
                a1.plot(df['KP'], df[COL_LIST[item]], linestyle = LINESTYLE_LIST[item],
                        label = f"{COL_LIST[item]} Comb. [Prob. {prob:.1f}%]")

            # Plot route bend, sleeper and ILT
            plot_bend_sleeper_ilt(a1, df1, df2, df3, df['1st Freq.'].max())

            a1.set_xlabel('KP (m)')
            a1.set_ylabel('Number of Buckles')
            a1.legend()
            a1.grid()

            plt.savefig(f'{WORK_DIR}\\zplot_{PIPELINE_ID}_comb_per_section_scen[{sce_no}].tiff',
                        dpi = dpi_size)
            if DISPLAY_PLOTS:
                plt.show()
            plt.close()

    if PLOT_ALL_SCENARIOS:

        # Generate matplolib figure
        fig = plt.figure()
        dpi_size = 110
        fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

        # Plot distribution of number of buckles
        a1 = fig.add_subplot(111)
        for sce_no in SCENARIO_LIST:
            df = df_dict[f"df_comb_section_{sce_no}"]

            # Plot the probability of the buckling combination
            # for item in range(df.filter(like = 'Prob').shape[1]):
            for item in range(1):
                prob = df.loc[1, f'Prob. - {COL_LIST[item]}'] * 100.0
                a1.plot(df['KP'], df[COL_LIST[item]], linestyle = LINESTYLE_LIST[item],
                        label = f"Scenario {sce_no} {COL_LIST[item]} Comb. [Prob. {prob:.1f}%]")

        # Plot route bend, sleeper and ILT
        df1 = df_dict[f"df_bend_{sce_no}"]
        df2 = df_dict[f"df_sleeper_{sce_no}"]
        df3 = df_dict[f"df_ilt_{sce_no}"]
        plot_bend_sleeper_ilt(a1, df1, df2, df3, df['1st Freq.'].max())

        a1.set_xlabel('KP (m)')
        a1.set_ylabel('Number of Buckles')
        a1.legend()
        a1.grid()

        plt.savefig(
            f'{WORK_DIR}\\zplot_{PIPELINE_ID}_comb_per_section_BuckPy_scen{SCENARIO_LIST}.tiff',
            dpi = dpi_size)
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()

#################################################
# Main
#################################################

# Obtain file names
WORK_DIR = r'C:\Github_repos\BuckPy\source'   # Change to the BuckPy working directory
BUCKPY_INPUT_FILE = 'inputDataTemplateA'      # Change to the BuckPy Excel template file
PIPELINE_LIST = ['A']                         # Pipeline IDs, ['A', 'B']
SCENARIO_LISTS = [[1]]                        # Scenario numbers for each pipeline, [[1,2], [1,2]]

# Write Buckfast's results to Excel
WRITE_BUCKFAST_EXCEL = True # True or False

# Plot Buckfast and BuckPy results for the current and/or all scenarios
PLOT_RESULTS = True # True or False
PLOT_COMBINATION = False # True or False, only True if output combination results in BuckPy

PLOT_CURRENT_SCENARIO = True # True or False, separate plots for each individual scenario
PLOT_ALL_SCENARIOS = True # True or False, combined plots showing all scenarios
DISPLAY_PLOTS = False # True or False, display plots on screen during execution

# Create summary table of Buckfast and BuckPy results for all scenarios
SUMMARY_TABLE = True # True or False

# Create a dictionary to store the Dataframes of all the pipelines
df_dict_summary = {}

# Define column name and line style list
COL_LIST = ['1st Freq.', '2nd Freq.', '3rd Freq.']
LINESTYLE_LIST = ['dashed', 'dotted', 'dashdot']

for index, PIPELINE_ID in enumerate(PIPELINE_LIST):

    # Create a dictionary to store the Dataframes of all scenarios
    df_dict_sce = {}

    SCENARIO_LIST = SCENARIO_LISTS[index]
    for SCENARIO in SCENARIO_LIST:

        if WRITE_BUCKFAST_EXCEL:

            print(f'Writing Buckfast result Excel for pipeline {PIPELINE_ID} scenario {SCENARIO}')

            # Find the line number of the df from the Buckfast result file
            elements_start, elements_end, set_prob_start, set_prob_end, no_buckle_start,\
                no_buckle_end, force_prof_start, force_prof_end, pipeline_start, pipeline_end = \
                    find_buckfast_line_no()

            # Create the df of the 'Elements' Tab in BuckPy results
            df_elem_buckfast = create_buckfast_tab_elements(elements_start, elements_end)

            # Create the df of the 'Sets' Tab in BuckPy results
            df_set_buckfast = create_buckfast_tab_sets(set_prob_start, set_prob_end)

            # Create the df of the 'No Buckles' Tab in BuckPy results
            df_buckle_buckfast = create_buckfast_tab_no_buckles(no_buckle_start, no_buckle_end)

            # Create the df of the 'Force Profiles' Tab in BuckPy results
            df_force_buckfast = create_buckfast_tab_force_prof(force_prof_start, force_prof_end)

            # Write the Buckfast result df into an output excel
            save_buckfast_output_file(df_elem_buckfast, df_set_buckfast,
                                      df_buckle_buckfast, df_force_buckfast)

        if PLOT_RESULTS | SUMMARY_TABLE:

            # Read the BuckPy and Buckfast result excel file
            if PLOT_COMBINATION:
                df_elem_buckpy, df_set_buckpy, df_buckle_buckpy, df_force_buckpy, *df_comb = \
                    read_output_file(f'{BUCKPY_INPUT_FILE}_{PIPELINE_ID}_scen{SCENARIO}_outputs')
                df_comb_per_set, df_comb_per_section = df_comb[0], df_comb[1]
            else:
                df_elem_buckpy, df_set_buckpy, df_buckle_buckpy, df_force_buckpy = \
                    read_output_file(f'{BUCKPY_INPUT_FILE}_{PIPELINE_ID}_scen{SCENARIO}_outputs')
            df_elem_buckfast, df_set_buckfast, df_buckle_buckfast, df_force_buckfast = \
                read_output_file(f'buckfast_{PIPELINE_ID}_scen{SCENARIO}')

            # Assembly a dataframe with the set results from BuckPy and Buckfast simulations
            df_set = assembly_dataframe_plot(df_set_buckpy, df_set_buckfast)

            # Assembly dataframes with the location of sleepers, ILTs and route bends
            df_bend, df_sleeper, df_ilt = assembly_dataframe_bend_sleeper_ilt()

            # Assembly dataframes with the buckling combination per set and per section
            if PLOT_COMBINATION:
                df_comb_per_set, df_comb_per_section = \
                    assembly_dataframe_combination_plot(df_comb_per_set, df_comb_per_section)

                # Store the Dataframes of all scenarios into a dictionary
                df_dict_sce = assembly_dataframe_scenarios(
                    df_dict_sce, df_set_buckpy, df_buckle_buckpy, df_force_buckpy, df_set_buckfast,
                    df_buckle_buckfast, df_force_buckfast, df_set, df_bend, df_sleeper, df_ilt,
                    df_comb_per_set, df_comb_per_section)
            else:
                # Store the Dataframes of all scenarios into a dictionary
                df_dict_sce = assembly_dataframe_scenarios(
                    df_dict_sce, df_set_buckpy, df_buckle_buckpy, df_force_buckpy, df_set_buckfast,
                    df_buckle_buckfast, df_force_buckfast, df_set, df_bend, df_sleeper, df_ilt)

    if PLOT_RESULTS:

        print(f'Plotting results for pipeline {PIPELINE_ID}')

        # Plot the unbuckled and buckled effective axial force profiles
        plot_force_profiles(df_dict_sce, 'EAF Operation [without Buckling] (kN)', 'unbuckled')
        plot_force_profiles(df_dict_sce, 'EAF Operation (kN)', 'buckled')

        # Plot the probability density function of the number of buckles
        plot_no_buckles(df_dict_sce)

        # Plot the probability of buckling per kilometer
        plot_prob_buckling(df_dict_sce)

        # Plot the characteristic VAS per kilometer
        plot_vas(df_dict_sce)

        # Plot the characteristic friction per kilometer
        plot_friction(df_dict_sce)

        if PLOT_COMBINATION:
            # Plot the number of buckles per kilometer for the first 3 most frequent set combination
            plot_comb_per_set(df_dict_sce)
            plot_comb_per_section(df_dict_sce)

    if SUMMARY_TABLE:
        # Obtain the dataframes of buckling and characteristic VAS for all the sceinarios
        df_buckling, df_char_vas = assembly_dataframe_summary(df_dict_sce)

        # Store the two df to a summary table dictionary
        df_dict_summary[f"df_buckling_{PIPELINE_ID}"] = df_buckling
        df_dict_summary[f"df_char_vas_{PIPELINE_ID}"] = df_char_vas

if SUMMARY_TABLE:
    print('Writing results into a summary table file.')

    # Concatenate summary tables of pipelines
    df_buckling_all, df_char_vas_all = assembly_dataframe_concat_summary(df_dict_summary)

    # Write the summary table dictionary to the excel
    save_summary_table_file(df_buckling_all, df_char_vas_all)
