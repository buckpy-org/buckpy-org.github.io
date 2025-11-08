'''
Script to convert the Buckfast output file into the BuckPy output file format
'''

import sys
import pandas as pd
import pandas.io.formats.excel
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import addcopyfighandler

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
    formatc1 = workbook.add_format({'num_format': '#,###', 'align': 'center',
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
    '''

    # Read all tabs in the Buckfast or BuckPy result file
    all_sheets_dict = pd.read_excel(f'{WORK_DIR}\\{result_file_name}.xlsx',
                                    sheet_name = None)

    # Read 'Elements', 'Sets', 'No Buckles' and 'Force Profiles' tab
    df_elem = all_sheets_dict['Elements']
    df_set = all_sheets_dict['Sets']
    df_buckle = all_sheets_dict['No Buckles']
    df_force_prof = all_sheets_dict['Force Profiles']

    return df_elem, df_set, df_buckle, df_force_prof

def assembly_dataframe_plot(df_buckpy, df_buckfast):

    """
    Plot probabilistic results of the number of buckles from Buckfast and BuckPy.

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

    # Double each row of the df
    df_temp1 = df.copy()
    df_temp1 = df_temp1.rename(columns = {'KP To (m)': 'KP'})
    df_temp1 = df_temp1.drop(labels = 'KP From (m)', axis = 1)
    df_temp1['sort'] = 1

    df_temp2 = df.copy()
    df_temp2 = df_temp2.rename(columns = {'KP From (m)': 'KP'})
    df_temp2 = df_temp2.drop(labels = 'KP To (m)', axis = 1)
    df_temp2['sort'] = 2

    df = pd.concat([df_temp1, df_temp2])
    df = df.sort_values(by = ['KP', 'sort'])
    df = df.reset_index(drop = True)

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
    df = pd.concat([df_temp1, df, df_temp2])
    df = df.fillna(0.0)

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

def plot_force_profiles(df_buckpy, df_buckfast, column_name, file_string):

    """
    Plot probabilistic results of the unbuckled and buckled EAF from Buckfast and BuckPy.

    Parameters
    ----------
    df_buckpy : pandas DataFrame
        The effective axial force in the BuckPy result.
    df_buckfast : pandas DataFrame
        The effective axial force in the Buckfast result.
    column_name : String
        The EAF column name in the dataframe.
    file_string : String
        'unbuckled' or 'buckled' conditions.
    """

    # Generate matplolib figure
    fig = plt.figure()
    dpi_size = 110
    fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

    # Plot distribution of number of buckles
    a1 = fig.add_subplot(111)
    a1.plot(df_buckpy['KP (m)'], df_buckpy[column_name],
            label = 'BuckPy')
    a1.plot(df_buckfast['KP (m)'], df_buckfast[column_name],
            label = 'Buckfast', linestyle = 'dashed')

    a1.set_xlabel('KP (m)')
    a1.set_ylabel('EAF (kN)')
    a1.legend()
    a1.grid()

    plt.savefig(f"{WORK_DIR}\\zplot_{PIPELINE_ID}_scen{SCENARIO}_force_profiles_{file_string}.tiff",
                dpi = dpi_size)
    if DISPLAY_PLOTS:
        plt.show()

def plot_no_buckles(df_buckpy, df_buckfast):

    """
    Plot probabilistic results of the number of buckles from Buckfast and BuckPy.

    Parameters
    ----------
    df_buckpy : pandas DataFrame
        Probability of number of buckles over pipeline in the BuckPy result.
    df_buckfast : pandas DataFrame
        Probability of number of buckles over pipeline in the Buckfast result.
    """

    # Generate matplolib figure
    fig = plt.figure()
    dpi_size = 110
    fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

    # Plot distribution of number of buckles
    a1 = fig.add_subplot(111)
    a1.plot(df_buckpy['Number of Buckles'], df_buckpy['Probability of Buckling'],
            label = 'BuckPy')
    a1.plot(df_buckfast['Number of Buckles'], df_buckfast['Probability of Buckling'],
            label = 'Buckfast', linestyle = 'dashed')

    a1.set_xlabel('Number of Buckles')
    a1.set_ylabel('Probability Density Function')
    a1.legend()
    a1.grid()

    plt.savefig(f"{WORK_DIR}\\zplot_{PIPELINE_ID}_scen{SCENARIO}_no_buckles.tiff",
                dpi = dpi_size)
    if DISPLAY_PLOTS:
        plt.show()

def plot_prob_buckling(df, df1, df2, df3):

    """
    Plot probabilities of buckling per kilometre.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the outputs from Buckfast and BuckPy simulations.
    df1 : pandas DataFrame
        Dataframe containing the locations of the route bends.
    df2 : pandas DataFrame
        Dataframe containing the locations of sleepers.
    df3 : pandas DataFrame
        Dataframe containing the locations of ILTs.
    """

    # Generate matplolib figure
    fig = plt.figure()
    dpi_size = 110
    fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

    # Plot distribution of number of buckles
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

    plt.savefig(f'{WORK_DIR}\\zplot_{PIPELINE_ID}_scen{SCENARIO}_prob_buckling.tiff',
                dpi = dpi_size)
    if DISPLAY_PLOTS:
        plt.show()

def plot_vas(df, df1, df2, df3):

    """
    Plot characteristic VAS per kilometre.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the outputs from Buckfast and BuckPy simulations.
    df1 : pandas DataFrame
        Dataframe containing the locations of the route bends.
    df2 : pandas DataFrame
        Dataframe containing the locations of sleepers.
    df3 : pandas DataFrame
        Dataframe containing the locations of ILTs.
    """

    # Generate matplolib figure
    fig = plt.figure()
    dpi_size = 110
    fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

    # Plot distribution of number of buckles
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

    plt.savefig(f'{WORK_DIR}\\zplot_{PIPELINE_ID}_scen{SCENARIO}_vas.tiff',
                dpi = dpi_size)
    if DISPLAY_PLOTS:
        plt.show()

def plot_friction(df, df1, df2, df3):

    """
    Plot characteristic friction per kilometre.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the outputs from Buckfast and BuckPy simulations.
    df1 : pandas DataFrame
        Dataframe containing the locations of the route bends.
    df2 : pandas DataFrame
        Dataframe containing the locations of sleepers.
    df3 : pandas DataFrame
        Dataframe containing the locations of ILTs.
    """

    # Generate matplolib figure
    fig = plt.figure()
    dpi_size = 110
    fig.set_size_inches(20.0 / 2.54, 12.0 / 2.54)

    # Plot distribution of number of buckles
    a1 = fig.add_subplot(111)
    a1.plot(df['KP'], df['Characteristic Lateral Breakout Friction, Buckles'],
            label = 'Characteristic Friction, Buckles')
    a1.plot(df['KP'], df['Lateral Breakout Friction, HE, Geotech'],
            label = 'Geotechnical Friction, HE', linestyle = 'dashed')

    # Plot route bend, sleeper and ILT
    plot_bend_sleeper_ilt(a1, df1, df2, df3,
                          df['Characteristic Lateral Breakout Friction, Buckles'].max())

    a1.set_xlabel('KP (m)')
    a1.set_ylabel('Breakout Lateral Friction')
    a1.legend()
    a1.grid()

    plt.savefig(f'{WORK_DIR}\\zplot_{PIPELINE_ID}_scen{SCENARIO}_friction.tiff',
                dpi = dpi_size)
    if DISPLAY_PLOTS:
        plt.show()

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

#################################################
# Main
#################################################

# Obtain file names
WORK_DIR = r'C:\Users\ismael.ripoll\Xodus Group\L200841-S00 - Documents\Simulation\PBLA'
BUCKPY_INPUT_FILE = 'buckpy'
PIPELINE_ID = 'PR'
SCENARIO = 1

# Write Buckfast's results to Excel
WRITE_BUCKFAST_EXCEL = True # True or False

# Plot Buckfast and BuckPy results
PLOT_RESULTS = True # True or False
DISPLAY_PLOTS = True # True or False

if WRITE_BUCKFAST_EXCEL:

    # Find the line number of the df from the Buckfast result file
    elements_start, elements_end, set_prob_start, set_prob_end, no_buckle_start, no_buckle_end,\
        force_prof_start, force_prof_end, pipeline_start, pipeline_end = find_buckfast_line_no()

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

if PLOT_RESULTS:

    # Read the BuckPy and Buckfast result excel file
    df_elem_buckpy, df_set_buckpy, df_buckle_buckpy, df_force_buckpy = \
        read_output_file(f'buckpy_{PIPELINE_ID}_scen{SCENARIO}_outputs')
    df_elem_buckfast, df_set_buckfast, df_buckle_buckfast, df_force_buckfast = \
        read_output_file(f'buckfast_{PIPELINE_ID}_scen{SCENARIO}')

    # Assembly a dataframe with the set results from BuckPy and Buckfast simulations
    df_set = assembly_dataframe_plot(df_set_buckpy, df_set_buckfast)

    # Assembly dataframes with the location of sleepers, ILTs and route bends
    df_bend, df_sleeper, df_ilt = assembly_dataframe_bend_sleeper_ilt()

    # Plot the unbuckled and buckled effective axial force profiles
    plot_force_profiles(df_force_buckpy, df_force_buckfast,
                        'EAF Operation [without Buckling] (kN)', 'unbuckled')
    plot_force_profiles(df_force_buckpy, df_force_buckfast,
                        'EAF Operation (kN)', 'buckled')

    # Plot the probability density function of the number of buckles
    plot_no_buckles(df_buckle_buckpy, df_buckle_buckfast)

    # Plot the probability of buckling per kilometer
    plot_prob_buckling(df_set, df_bend, df_sleeper, df_ilt)

    # Plot the characteristic VAS per kilometer
    plot_vas(df_set, df_bend, df_sleeper, df_ilt)

    # Plot the characteristic friction per kilometer
    plot_friction(df_set, df_bend, df_sleeper, df_ilt)
