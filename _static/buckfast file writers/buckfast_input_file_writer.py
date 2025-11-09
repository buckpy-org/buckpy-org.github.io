'''
Script to create the Buckfast input file from the BuckPy input file
and autorun the Buckfast for all the pipelines and scenarios.
Note: need to run the script in the folder where buckfast132.exe and input csv files exist.
'''

import sys
import pandas as pd
import numpy as np
sys.path.append(r'C:\Github_repos\BuckPy\source')
from buckpy_preprocessing import LBDistributions

def calc_inner_diameter(od, wt):

    """
    Calculate the pipe inner diameter

    Parameters
    ----------
    od : float
        Outer diameter
    wt : float
        Wall thickness

    Returns
    -------
    inner_diameter : float
        Pipe inner diameter
    """

    inner_diameter = od - 2.0 * wt

    return inner_diameter

def calc_area_inner(od, wt):

    """
    Calculate the pipe inner area

    Parameters
    ----------
    od : float
        Outer diameter
    wt : float
        Wall thickness

    Returns
    -------
    area_inner : float
        Pipe inner area
    """

    inner_diameter = calc_inner_diameter(od, wt)
    area_inner = np.pi / 4.0 * inner_diameter**2

    return area_inner

def calc_area_outer(od):

    """
    Calculate the pipe outer area

    Parameters
    ----------
    od : float
        Outer diameter

    Returns
    -------
    area_outer : float
        Pipe outer area
    """

    area_outer = np.pi / 4.0 * od**2

    return area_outer

def calc_area_steel(od, wt):

    """
    Calculate the steel cross-sectional area

    Parameters
    ----------
    od : float
        Outer diameter
    wt : float
        Wall thickness

    Returns
    -------
    area_steel : float
        Steel cross-sectional area
    """

    area_outer = calc_area_outer(od)
    area_inner = calc_area_inner(od, wt)
    area_steel = area_outer - area_inner

    return area_steel

def calc_area_tod(od, coat_wt):

    """
    Calculate the outer area of steel and coating

    Parameters
    ----------
    od : float
        Steel outer diameter
    coat_wt : float
        Coating wall thickness

    Returns
    -------
    area_tod : float
        Outer area of steel and coating
    """

    tod = od + 2.0 * coat_wt
    area_tod = np.pi / 4.0 * tod**2

    return area_tod

def calc_coat_area(od, coat_wt):

    """
    Calculate the coating cross-sectional area

    Parameters
    ----------
    od : float
        Steel outer diameter
    coat_wt : float
        Coating wall thickness

    Returns
    -------
    coat_area : float
        Coating cross-sectional area
    """

    area_tod = calc_area_tod(od, coat_wt)
    area_outer = calc_area_outer(od)
    coat_area = area_tod - area_outer

    return coat_area

def calc_ax_stiff(od, wt, young_modulus):

    """
    Calculate the axial stiffness

    Parameters
    ----------
    od : float
        Outer diameter
    wt : float
        Wall thickness
    young_modulus : float
        Young's modulus

    Returns
    -------
    axial_stiffness : float
        Axial stiffness
    """

    area_steel = calc_area_steel(od, wt)
    axial_stiffness = young_modulus * area_steel

    return axial_stiffness

def calc_area_moment_inertia(od, wt):

    """
    Calculate the area moment inertia

    Parameters
    ----------
    od : float
        Outer diameter
    wt : float
        Wall thickness

    Returns
    -------
    area_moment_inertia : float
        Area moment inertia
    """

    inner_diameter = calc_inner_diameter(od, wt)
    area_moment_inertia = np.pi / 64.0 * (od**4 - inner_diameter**4)

    return area_moment_inertia

def calc_bend_stiff(od, wt, young_modulus):

    """
    Calculate the bending stiffness

    Parameters
    ----------
    od : float
        Outer diameter
    wt : float
        Wall thickness
    young_modulus : float
        Young's modulus

    Returns
    -------
    bending_stiffnes : float
        Bending stiffness
    """

    area_moment_inertia = calc_area_moment_inertia(od, wt)
    bending_stiffnes = young_modulus * area_moment_inertia

    return bending_stiffnes

def calc_cbf(route_type, loc_fric, scale_fric, loc_oos, scale_oos, ea, ei, sw, radius, h, rcm_force):

    '''
    Compute the parameters of a lognormal distribution for CBF (STRAIGHT, BEND, SLEEPER or RCM sections).

    Parameters
    ----------
    route_type : string
        The route type: 'STRAIGHT', 'BEND', 'SLEEPER' or 'RCM'
    loc_fric : float
        The location parameter of the lognormal friction distribution
    scale_fric : float
        The scale paramater of the lognormal friction distribution
    loc_OOS : float
        The location parameter of the lognormal OOS distribution
    scale_OOS : float
        The scale paramater of the lognormal OOS distribution
    ea : float
        Axial stiffness
    ei : float
        Bending stiffness
    sw : float
        Submerged weight
    radius : float
        Bend radius
    h : float
        Sleeper height
    rcm_force : float
        RCM buckling force

    Returns
    -------
    loc_cbf : float
        Location parameter of the lognormal CBF distribution
    scale_cbf : float
        scale paramater of the lognormal CBF distribution
    mean_cbf : float
        Mean of the lognormal CBF distribution
    std_cbf : float
        Standard deviation of the lognormal CBF distribution
    '''

    # Calculate loc_cbf and scale_cbf
    if route_type == 'STRAIGHT':
        s_char = 2.26 * ea**0.25 * ei**0.25 * sw**0.5
        loc_cbf = 0.5 * loc_fric + loc_oos + np.log(s_char)
        scale_cbf = np.sqrt((0.5 * scale_fric)**2 + scale_oos**2)

    elif route_type == 'BEND':
        loc_cbf = loc_fric + loc_oos + np.log(int(radius)) + np.log(sw)
        scale_cbf = np.sqrt(scale_fric**2 + scale_oos**2)

    elif route_type == 'SLEEPER':
        prop_uhb = 4.0 * np.sqrt(ei * sw / h)
        loc_cbf = loc_oos + np.log(prop_uhb)
        scale_cbf = scale_oos

    elif route_type == 'RCM':
        loc_cbf = loc_oos + np.log(rcm_force)
        scale_cbf = scale_oos

    else:
        loc_cbf = loc_oos
        scale_cbf = scale_oos

    # Calculate mean_cbf and std_cbf
    if route_type == 'SLEEPER':
        mean_oos= np.exp(loc_oos + scale_oos**2 / 2)
        std_oos = np.sqrt((np.exp(scale_oos**2) - 1) * np.exp(2 * loc_oos + scale_oos**2))
        mean_cbf = mean_oos * prop_uhb
        std_cbf = std_oos * prop_uhb

    else:
        mean_cbf = np.exp(loc_cbf + scale_cbf**2 / 2)
        std_cbf = np.sqrt((np.exp(scale_cbf**2) - 1) * np.exp(2 * loc_cbf + scale_cbf**2))

    return loc_cbf, scale_cbf, mean_cbf, std_cbf

def calc_expand_kp(df, kp_col):

    '''
    Function to expand the KP array with 1000 intervals from 1000 to nearest maximum KP.

    Parameters
    ----------
    df : pandas Dataframe
        Dataframe containing the original KP values.
    kp_col : string
        The column name of the KP values to expand.

    Returns
    -------
    df : pandas Dataframe
        Dataframe containing the expanded KP values.
    '''

    # Rename kp_col to 'KP From'
    df = df.rename(columns = {kp_col: 'KP From'})

    # Expand the KP array with 1000 intervals from 1000 to nearest maximum KP
    max_kp = np.floor(df['KP From'].max() / 1000.0) * 1000.0
    kp_array = np.arange(1000, max_kp + 1.0, 1000)

    # Create a dataframe for the expanded kp
    df_expand = pd.DataFrame({'Point ID From': [np.NAN] * len(kp_array), 'KP From': kp_array})
    df = pd.concat([df, df_expand], ignore_index = True).sort_values(
        by = 'KP From').drop_duplicates('KP From').reset_index(drop = True).ffill()

    # Calculate relative length between KP and KP To
    df['KP To'] = df['KP From'].shift(-1)
    df = df.dropna()
    df['Length'] = df['KP To'] - df['KP From']

    # Calculate element number and element size
    df['Elem No.'] = np.ceil(df['Length'] / 100.0)
    df['Elem Size'] = df['Length'] / df['Elem No.']

    return df

def calc_element_array(df):

    '''
    Function to create element array based on KP, KP TO and element number.

    Parameters
    ----------
    df : pandas Dataframe
        Dataframe containing the expanded KP values.

    Returns
    -------
    df : pandas Dataframe
        Dataframe containing the elements between each KP value.
    '''

    # Create the elements between each KP points
    elem_array = np.empty(0)
    elem_array = df.apply(lambda x: pd.Series(np.append(elem_array, np.linspace(
        x['KP From'], x['KP To'], int(x['Elem No.'] + 1.0)))), axis = 1)

    # Convert the element dataframe to np array and flatten
    elem_array = elem_array.to_numpy().flatten()

    # Remove duplicated values at 1000*n and np.nan
    elem_array = np.unique(elem_array)
    elem_array = elem_array[~np.isnan(elem_array)]

    return elem_array

def calc_kp_interpolation(elem_array, df_oper):

    '''
    Function to interpolate the RLT, pressure and temperature using KP and operating profile.

    Parameters
    ----------
    elem_array : np Array
        Array containing the kp value of the elements.
    df_oper : pandas Dataframe
        Dataframe containing the original operating profiles data.

    Returns
    -------
    df : pandas Dataframe
        Dataframe containing the interpolated operating profiles data.
    '''

    # Interpolate operating profile based on KP
    df = pd.DataFrame({'KP': elem_array})
    df['Pressure Installation'] = np.interp(
        df['KP'], df_oper['KP'], df_oper['Pressure Installation'])
    df['Pressure Hydrotest'] = np.interp(
        df['KP'], df_oper['KP'], df_oper['Pressure Hydrotest'])
    df['Pressure Operation'] = np.interp(
        df['KP'], df_oper['KP'], df_oper['Pressure Operation'])
    df['Temperature Installation'] = np.interp(
        df['KP'], df_oper['KP'], df_oper['Temperature Installation'])
    df['Temperature Hydrotest'] = np.interp(
        df['KP'], df_oper['KP'], df_oper['Temperature Hydrotest'])
    df['Temperature Operation'] = np.interp(
        df['KP'], df_oper['KP'], df_oper['Temperature Operation'])
    df['RLT'] = np.interp(df['KP'], df_oper['KP'], df_oper['RLT'])

    return df

def calc_operating_profiles(df, df_route, pipeline_set, loadcase_set):

    """
    Calculate operating profiles data and process it.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the operating profiles data.
    df_route : pandas.DataFrame
        DataFrame containing route data and calculated route data.
    pipeline_set : str
        Identifier of the pipeline set.
    loadcase_set : str
        Identifier of the loadcase set.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the operating profiles data and calculated operating data.
    """

    # Filter df DataFrame based on pipeline_set and loadcase_set
    df_profile = df.loc[(df['Pipeline'] == pipeline_set) & (df['Loadcase Set'] == loadcase_set)]

    # Select the 'Point ID From' and 'KP To' columns
    df_route = df_route[['Point ID From', 'KP To']].reset_index(drop = True)

    # Add the end row of route and the start KP
    end_row = pd.DataFrame({'Point ID From': 'End', 'KP To': np.nan}, index = [99999])
    df_route = pd.concat([df_route, end_row], ignore_index = True)

    # Shift KP column 1 downwards and assign 0.0 to the first KP
    df_route['KP To'] = df_route['KP To'].shift().fillna(0.0)

    # Expand the KP array with 1000 intervals from 1000 to nearest maximum KP
    df_route = calc_expand_kp(df_route, 'KP To')

    # Create the elements between each KP points
    elem_array = calc_element_array(df_route)

    # Interpolate the RLT, pressure and temperature using KP and operating profile
    df = calc_kp_interpolation(elem_array, df_profile)

    # Insert pipeline_set and loadcase_set columns as the first and second columns
    df.insert(0, 'Pipeline', [pipeline_set] * df.shape[0])
    df.insert(1, 'Loadcase Set', [loadcase_set] * df.shape[0])

    return df

def calc_densities(od, wt, coat_wt, steel_density, water_density, sw_empty, sw_inst, sw_oper):

    '''
    Calculate parameters of coating density, content density during operation and installation.

    Parameters
    ----------
    od : float
        Outer diameter
    wt : float
        Wall thickness
    coat_wt : float
        Coating wall thickness
    steel_density : float
        Steel density
    water_density : float
        Seawater density
    sw_inst : float
        Submerged weight during installation
    sw_oper : float
        Submerged weight during operation

    Returns
    -------
    coat_density : float
        Coating density
    inst_density : float
        Content density during installation
    oper_density : float
        Content density during operation
    '''

    # Calculate area_inner, area_steel, area_tod and coat_area based on od, wt, coat_wt
    area_inner = calc_area_inner(od, wt)
    area_steel = calc_area_steel(od, wt)
    area_tod = calc_area_tod(od, coat_wt)
    coat_area = calc_coat_area(od, coat_wt)

    # Calculate coat_density, inst_density and oper_density based on the given coat_wt,
    #   steel_density and water_density using the sw equations
    coat_density = (sw_empty / 9.807 - (
        steel_density * area_steel - water_density * area_tod)) / coat_area
    inst_density = (sw_inst - sw_empty) / 9.807 / area_inner
    oper_density = (sw_oper - sw_empty) / 9.807 / area_inner

    # The coating density should not be much smaller than sea water
    if coat_density < 0.0:
        print("The coating density is negative. Change the COAT_WT.")
        quit()

    return coat_density, inst_density, oper_density

def calc_sets(df, index_max):

    '''
    Calculate the length and number of elements of each element set,
    and calculate the property sets.

    Parameters
    ----------
    df : Dataframe
        Dataframe containing data from the BuckPy input file.
    index_max : int
        Largest index of the Dataframe containing data from
        the BuckPy input file.

    Returns
    -------
    df : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Calculate length and element number based on KP and index
    df['KP Shift'] = df['KP'].shift(-1).fillna(df['KP To'].max())
    df['Length'] = df['KP Shift'] - df['KP']
    df['index_copy Shift'] = df['index_copy'].shift(-1).fillna(index_max)
    df['No. Elements'] = df['index_copy Shift'] - df['index_copy']

    # Calculate element and property set
    df = df.reset_index(drop = True).reset_index()
    df['Property Set'] = df['index'] + 1

    return df

def calc_cbf_mean_std(route_type, od, wt, young_modulus, radius, height,
            rcm, sw_hydr, sw_oper, mean_lat_hydr, std_lat_hydr,
            mean_lat_oper, std_lat_oper, mean_oos, std_oos):

    '''
    Calculate the mean and std of the lognormal CBF distritution 
    during hydrotest and operation.

    Parameters
    ----------
    route_type : string
        Section type: 'STRAIGHT' or 'BEND' or 'SLEEPER'
    od : float
        Outer diameter
    wt : float
        Wall thickness
    young_modulus : float
        Young's modulus
    radius : float
        Bend radius
    height : float
        Sleeper height
    rcm : float
        RCM buckling force
    sw_hydr : float
        Submerged weight during hydrotest
    sw_oper : float
        Submerged weight during operation
    mean_lat_hydr : float
        Mean parameter of the lognormal friction distribution during hydrotest
    std_lat_hydr : float
        Standard deviation paramater of the lognormal friction distribution during hydrotest
    mean_lat_oper : float
        Mean parameter of the lognormal friction distribution during operation
    std_lat_oper : float
        Standard deviation paramater of the lognormal friction distribution during operation
    mean_oos : float
        Mean parameter of the lognormal OOS distribution
    std_oos : float
        Standard deviation paramater of the lognormal OOS distribution

    Returns
    -------
    mean_cbf_hydr : float
        Mean of the lognormal CBF distribution during hydrotest
    std_cbf_hydr : float
        Standard deviation of the lognormal CBF distribution during hydrotest
    mean_cbf_oper : float
        Mean of the lognormal CBF distribution during operation
    std_cbf_oper : float
        Standard deviation of the lognormal CBF distribution during operation
    '''

    # Calculate EA and EI from od, wt and E
    ax_stiff = calc_ax_stiff(od, wt, young_modulus)
    bend_stiff = calc_bend_stiff(od, wt, young_modulus)

    # Calculate loc and scale from mean and std of OOS
    loc_oos = np.log(mean_oos**2 / np.sqrt(mean_oos**2 + std_oos**2))
    scale_oos = np.sqrt(np.log(1 + std_oos**2 / mean_oos**2))

    # Calculate loc and scale from mean and std of hydrotest friction
    loc_lat_hydr = np.log(mean_lat_hydr**2 / np.sqrt(mean_lat_hydr**2 + std_lat_hydr**2))
    scale_lat_hydr = np.sqrt(np.log(1 + std_lat_hydr**2 / mean_lat_hydr**2))

    # Calculate loc and scale from mean and std of operating friction
    loc_lat_oper = np.log(mean_lat_oper**2 / np.sqrt(mean_lat_oper**2 + std_lat_oper**2))
    scale_lat_oper = np.sqrt(np.log(1 + std_lat_oper**2 / mean_lat_oper**2))

    # Calculate mean and std
    loc_cbf_hydr, scale_cbf_hydr, mean_cbf_hydr, std_cbf_hydr = calc_cbf(
		route_type, loc_lat_hydr, scale_lat_hydr, loc_oos,
        scale_oos, ax_stiff, bend_stiff, sw_hydr, radius, height, rcm)
    loc_cbf_oper, scale_cbf_oper, mean_cbf_oper, std_cbf_oper = calc_cbf(
        route_type, loc_lat_oper, scale_lat_oper, loc_oos,
        scale_oos, ax_stiff, bend_stiff, sw_oper, radius, height, rcm)

    return mean_cbf_hydr, std_cbf_hydr, mean_cbf_oper, std_cbf_oper

def open_file():

    '''
    Read the data from the BuckPy input file.

    Parameters
    ----------
    BUCKPY_EXCEL : string
        File path of the BuckPy input file.
    SCENARIO_NO : int
        Scenario number of current simulation.

    Returns
    -------
    df : Dataframe
        Dataframe containing data from all the tabs in the BuckPy input file.
    df_route_ends : Dataframe
        Dataframe containing data from layout ends.
    '''

    # Read all tabs in the BuckPy input file
    all_sheets_dict = pd.read_excel(rf'{WORK_DIR}\{BUCKPY_EXCEL}', sheet_name = None)

    # Read 'Scenario' tab and select current scenario number
    df_sce = all_sheets_dict['Scenario']
    df_sce = df_sce.loc[(df_sce['Pipeline'] == PIPELINE_ID) &
                        (df_sce['Scenario'] == SCENARIO_NO)]
    loadcase_set = df_sce['Loadcase Set'].iloc[0]

    # Read 'Route' tab and convert KP from int to float to fix merge warning on dtype
    df_route = all_sheets_dict['Route']
    df_route[['KP From', 'KP To']] = df_route[['KP From', 'KP To']].astype(float)
    df_route['Route Type'] = df_route['Route Type'].str.upper()
    df_route = df_route.loc[(df_route['Pipeline'] == PIPELINE_ID) &
                            (df_route['Layout Set'].isin(df_sce['Layout Set']))]
    df_route_ends = df_route.loc[((df_route['Route Type'] == 'SPOOL') |
                                  (df_route['Route Type'] == 'FIXED'))]
    df_route = df_route.loc[~((df_route['Route Type'] == 'SPOOL') |
                              (df_route['Route Type'] == 'FIXED'))]

    # Read 'Pipe' tab
    df_pipe = all_sheets_dict['Pipe']
    df_pipe = df_pipe.loc[(df_pipe['Pipeline'] == PIPELINE_ID) &
                          (df_pipe['Pipe Set'].isin(df_route['Pipe Set']))]

    # Read 'Soils' tab
    df_soil = all_sheets_dict['Soils']
    df_soil = df_soil.loc[(df_soil['Pipeline'] == PIPELINE_ID) &
                          (df_soil['Friction Set'].isin(df_route['Friction Set']))]

    # Calculate the mean and std of soil friction
    # Axial
    df_soil['Axial Mean'], df_soil['Axial STD'] = LBDistributions(
        friction_factor_le=[df_soil['Axial LE']],
        friction_factor_be=[df_soil['Axial BE']],
        friction_factor_he=[df_soil['Axial HE']],
        friction_factor_fit_type=[df_soil['Axial Fit Bounds']]
    ).friction_distribution()[:2]
    # Lateral Hydrotest
    df_soil['Lateral Hydrotest Mean'], df_soil['Lateral Hydrotest STD'] = LBDistributions(
        friction_factor_le=[df_soil['Lateral Hydrotest LE']],
        friction_factor_be=[df_soil['Lateral Hydrotest BE']],
        friction_factor_he=[df_soil['Lateral Hydrotest HE']],
        friction_factor_fit_type=[df_soil['Lateral Hydrotest Fit Bounds']]
    ).friction_distribution()[:2]
    # Lateral Operation
    df_soil['Lateral Operation Mean'], df_soil['Lateral Operation STD'] = LBDistributions(
        friction_factor_le=[df_soil['Lateral Operation LE']],
        friction_factor_be=[df_soil['Lateral Operation BE']],
        friction_factor_he=[df_soil['Lateral Operation HE']],
        friction_factor_fit_type=[df_soil['Lateral Operation Fit Bounds']]
    ).friction_distribution()[:2]

    # Read 'Operating' tab and filter the operating profile for the current scenario
    df_oper = all_sheets_dict['Operating']
    df_oper = df_oper.loc[(df_oper['Pipeline'] == PIPELINE_ID) &
                          (df_oper['Loadcase Set'].isin(df_sce['Loadcase Set']))]
    df_oper = calc_operating_profiles(df_oper, df_route, PIPELINE_ID, loadcase_set)

    # Read 'Post-Processing' tab
    df_pp = all_sheets_dict['Post-Processing']
    df_pp = df_pp.loc[(df_pp['Pipeline'] == PIPELINE_ID) &
                      (df_pp['Layout Set'].isin(df_sce['Layout Set']))]
    df_pp = df_pp[['Pipeline', 'Layout Set', 'Post-Processing Set', 'KP From', 'KP To']]

    # Merge df_sce and df_route on 'Pipeline' and 'Layout Set'
    df = pd.merge(df_sce, df_route, on = ['Pipeline', 'Layout Set'], how = 'left')

    # Merge df and df_pipe on 'Pipeline' and 'Pipe Set'
    df = pd.merge(df, df_pipe, on = ['Pipeline', 'Pipe Set'], how = 'left')

    # Merge df and df_soil on 'Pipeline' and 'Friction Set'
    df = pd.merge(df, df_soil, on = ['Pipeline', 'Friction Set'], how = 'left')

    # Merge df and df_oper on 'Pipeline', 'Loadcase Set' and 'KP From'/'KP'
    df = pd.merge(df, df_oper, left_on = ['Pipeline', 'Loadcase Set', 'KP From'],
                  right_on = ['Pipeline', 'Loadcase Set', 'KP'], how = 'right')

    # Concatenate df and df_pp
    for index, row in df.iterrows():
        if df_pp.loc[df_pp['KP From'] == row['KP'], 'Post-Processing Set'].size > 0:
            df.loc[index, 'Element Set'] = df_pp.loc[df_pp['KP From'] == row['KP'],
                                                     'Post-Processing Set'].iloc[0]
            df.loc[index, 'Counter'] = index
        if row['KP From'] > 0.0:
            df.loc[index, 'Counter'] = index

    # Forward fill df
    df = df.ffill()

    # Add property values that do not change
    df['water_density'] = WATER_DENSITY
    df['axial_fric_dist'] = AXIAL_FRIC_DIST
    df['coat_wt'] = COAT_WT
    df['steel_density'] = STEEL_DENSITY
    df['coat_layer'] = COAT_LAYER
    df['cbf_no'] = CBF_NO
    df['RLT'] = abs(df['RLT'])

    # Re-assign to the sets ascending numbers starting from zero, considering duplicates
    df['Layout Set'] = df['Layout Set'].rank(method = 'dense').astype(int)
    df['Pipe Set'] = df['Pipe Set'].rank(method = 'dense').astype(int)
    df['Friction Set'] = df['Friction Set'].rank(method = 'dense').astype(int)
    df['Loadcase Set'] = df['Loadcase Set'].rank(method = 'dense').astype(int)

    return df, df_route_ends

def func_heading(df_in):

    '''
    Create the Buckfast input file and add the *HEADING section.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Create the Buckfast input file and add the *HEADING section
    full_description = (f"{DESCRIPTION} - Pipeline {df_in['Pipeline'].iloc[0]} - "
                        f"Scenario {int(df_in['Scenario'].iloc[0])}")
    df_out = pd.DataFrame({
        'Col_1': np.append('*HEADING', full_description)
        })

    return df_out

def func_simulations(df_in, df_out):

    '''
    Add the *SIMULATIONS section to the Buckfast input file.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Create a dataframe for the *SIMULATIONS section
    df_out_temp = pd.DataFrame({
        'Col_1': np.append('*SIMULATIONS', int(df_in['Simulations'].iloc[0]))
        })

    # Add the *SIMULATIONS section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)

    return df_out

def func_water(df_in, df_out):

    '''
    Add the *WATER section to the Buckfast input file.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Create a dataframe for the *WATER section
    df_out_temp = pd.DataFrame({
        'Col_1': np.append('*WATER', df_in['Temperature Installation'].iloc[0]),
        'Col_2': np.append(np.nan, df_in['water_density'].iloc[0])
        })

    # Add the *WATER section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)

    return df_out

def func_friction(df_in, df_out):

    '''
    Add the *AXIALFRICTION section to the Buckfast input file.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Select the unique 'Friction Set'
    df_in = df_in.dropna(subset = 'Friction Set')
    df_in = df_in.drop_duplicates(subset = 'Friction Set')

    # Create a dataframe for the title of the *AXIALFRICTION section
    df_out_title = pd.DataFrame({
        'Col_1': ['*AXIALFRICTION']
        })

    # Create a dataframe for the first line of the *AXIALFRICTION section
    df_out_first_line = pd.DataFrame({
        'Col_1': df_in['axial_fric_dist'],
        'Col_2': df_in['Axial Mean'],
        'Col_3': df_in['Axial STD'],
        'Col_4': df_in['Friction Set'].astype(int).astype(str)
        })

    # Create a dataframe for all the rows of the *AXIALFRICTION section
    df_out_temp = pd.concat([df_out_title, df_out_first_line])

    # Add the *AXIALFRICTION section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)

    return df_out

def func_single(df_in, df_out):

    '''
    Add the *SINGLE section to the Buckfast input file.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Select the unique 'Pipe Set'
    df_in = df_in.dropna(subset = 'Pipe Set')
    df_in = df_in.drop_duplicates(subset = 'Pipe Set')

    # Calculate the coating density, content density during hydrotest and operation
    df_in[['coat_density', 'inst_density', 'oper_density']] = df_in.apply(
        lambda row: pd.Series(calc_densities(
            row['OD'], row['WT'], row['coat_wt'], row['steel_density'], row['water_density'],
            row['sw Empty'], row['sw Installation'], row['sw Operation'])), axis = 1)

    # Use the equivalent pipe properties of a single pipe for pipe-in-pipe and convert unit to mm
    df_in[['OD', 'WT']] = df_in[['OD', 'WT']] * 1000.0
    df_in['E'] = df_in['E'] / 1.0E+06

    # Create a dataframe for the title of the *SINGLE section
    df_out_title = pd.DataFrame({
        'Col_1': ['*SINGLE']
        })

    # Create a dataframe for the first rows of the *SINGLE section
    df_out_first_line = pd.DataFrame({
        'Col_1': df_in['OD'],
        'Col_2': df_in['WT'],
        'Col_3': df_in['steel_density'],
        'Col_4': df_in['coat_layer'].astype(int).astype(str),
        'Col_5': df_in['E'],
        'Col_6': df_in['Poisson'],
        'Col_7': df_in['Alpha'],
        'Col_8': df_in['oper_density'],
        'Col_9': df_in['inst_density'],
        'Col_10': df_in['Pipe Set'].astype(int).astype(str)
        })

    # Create a dataframe for the second rows of the *SINGLE section
    df_out_second_line = pd.DataFrame({
        'Col_1': 1000.0 * df_in['coat_wt'],
        'Col_2': df_in['coat_density']
        })

    # Create a dataframe for all the rows of the *SINGLE section
    df_out_temp = pd.concat([df_out_title, df_out_first_line, df_out_second_line])
    df_out_temp = df_out_temp.sort_index()

    # Add the *SINGLE section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)

    return df_out

def func_installation(df_in, df_out):

    '''
    Add the *INSTALLATION section to the Buckfast input file.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Convert unit to N*mm^(-2)
    df_in.loc[:, 'Pressure Installation'] = df_in['Pressure Installation'] / 1.0E+06

    # Create a dataframe for the title of the *INSTALLATION section
    df_out_title = pd.DataFrame({
        'Col_1': ['*INSTALLATION']
        })

    # Create a dataframe for the first row of the *INSTALLATION section
    df_out_first_line = pd.DataFrame({
        'Col_1': ['LAYTENSION', 'PRESSURE', 'TEMPERATURE'],
        'Col_2': 'TABULAR',
        'Col_3': [df_in['RLT'].iloc[0], df_in['Pressure Installation'].iloc[0],
                  df_in['Temperature Installation'].iloc[0]],
        'Col_4': [df_in['RLT'].iloc[-1], df_in['Pressure Installation'].iloc[-1],
                  df_in['Temperature Installation'].iloc[-1]]
        })

    # Create a dataframe for all the rows of the *INSTALLATION section
    df_out_temp = pd.concat([df_out_title, df_out_first_line])

    # Add the *INSTALLATION section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)

    return df_out

def func_hydrotest(df_in, df_out):

    '''
    Add the *HYDROTEST section to the Buckfast input file.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Convert unit to N*mm^(-2)
    df_in.loc[:, 'Pressure Hydrotest'] = df_in['Pressure Hydrotest'] / 1.0E+06

    # Create a dataframe for the title of the *HYDROTEST section
    df_out_title = pd.DataFrame({
        'Col_1': ['*HYDROTEST']
        })

    # Create a dataframe for the first row of the *HYDROTEST section
    df_out_first_line = pd.DataFrame({
        'Col_1': ['PRESSURE', 'TEMPERATURE'],
        'Col_2': 'TABULAR',
        'Col_3': [df_in['Pressure Hydrotest'].iloc[0], df_in['Temperature Hydrotest'].iloc[0]],
        'Col_4': [df_in['Pressure Hydrotest'].iloc[-1], df_in['Temperature Hydrotest'].iloc[-1]]
        })

    # Create a dataframe for all the rows of the *HYDROTEST section
    df_out_temp = pd.concat([df_out_title, df_out_first_line])

    # Add the *HYDROTEST section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)

    return df_out

def func_operation(df_in, df_out):

    '''
    Add the *OPERATION section to the Buckfast input file.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Convert unit to N*mm^(-2)
    df_in.loc[:, 'Pressure Operation'] = df_in['Pressure Operation'] / 1.0E+06

    # Create a dataframe for the title of the *OPERATION section
    df_out_title = pd.DataFrame({
        'Col_1': ['*OPERATION']
        })

    # Create a dataframe for the first row of the *OPERATION section
    df_out_first_line = pd.DataFrame({
        'Col_1': ['PRESSURE', 'TEMPERATURE'],
        'Col_2': 'TABULAR',
        'Col_3': [df_in['Pressure Operation'].iloc[0], df_in['Temperature Operation'].iloc[0]],
        'Col_4': [df_in['Pressure Operation'].iloc[-1], df_in['Temperature Operation'].iloc[-1]]
        })

    # Create a dataframe for all the rows of the *OPERATION section
    df_out_temp = pd.concat([df_out_title, df_out_first_line])

    # Add the *OPERATION section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)

    return df_out

def func_route(df_in, df_out):

    '''
    Add the *ROUTE section to the Buckfast input file.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Copy the index of the entire input DataFrame
    df_in = df_in.reset_index()
    df_in['index_copy'] = df_in['index']
    df_in = df_in.drop(labels = 'index', axis = 1)

    # Select rows for the long lines of *ROUTE second line
    df_in_long_line = df_in.copy().drop_duplicates(subset = ['Counter'],
                                                   keep = 'first')

    index_list = df_in_long_line.index
    df_in_long_line = calc_sets(df_in_long_line, df_in['index_copy'].max())
    df_in_long_line_copy = df_in_long_line.copy()

    # Replace "RCM" in "Route Type" column to "SLEEPER"
    df_in_long_line['Route Type'] = df_in_long_line['Route Type'].replace("RCM", "SLEEPER")

    # Select rows for the short line of *ROUTE second line
    df_in_short_line = df_in.loc[~df_in['index_copy'].isin(index_list)].iloc[:-1]

    # Create a dataframe for the title of the *ROUTE section
    df_out_title = pd.DataFrame({
        'Col_1': ['*ROUTE']
        })

    # Create a dataframe for the first line of the *ROUTE section
    df_out_first_line = pd.DataFrame({
        'Col_1': ['END'],
        'Col_2': ['0'],
        'Col_3': ['1'],
        'Col_4': ['1'],
        'Col_5': [f"{df_in['Friction Set'].iloc[0]:.0f}"],
        'Col_6': [f"{df_in['Pipe Set'].iloc[0]:.0f}"],
        'Col_7': ['0'],
        'Col_8': [df_in['RLT'].iloc[0]],
        'Col_9': [df_in['Pressure Installation'].iloc[0]],
        'Col_10': [df_in['Temperature Installation'].iloc[0]],
        'Col_11': [df_in['Pressure Hydrotest'].iloc[0]],
        'Col_12': [df_in['Temperature Hydrotest'].iloc[0]],
        'Col_13': [df_in['Pressure Operation'].iloc[0]],
        'Col_14': [df_in['Temperature Operation'].iloc[0]]
        })

    # Create a dataframe for the first line of the *ROUTE section and use original index
    df_out_second_line_1 = pd.DataFrame({
        'Col_1': df_in_long_line['Route Type'],
        'Col_2': df_in_long_line['Length'].astype(int).astype(str),
        'Col_3': df_in_long_line['No. Elements'].astype(int).astype(str),
        'Col_4': df_in_long_line['Property Set'].astype(int).astype(str),
        'Col_5': df_in_long_line['Friction Set'].astype(int).astype(str),
        'Col_6': df_in_long_line['Pipe Set'].astype(int).astype(str),
        'Col_7': df_in_long_line['Element Set'].astype(int).astype(str),
        'Col_8': df_in_long_line['RLT'],
        'Col_9': df_in_long_line['Pressure Installation'],
        'Col_10': df_in_long_line['Temperature Installation'],
        'Col_11': df_in_long_line['Pressure Hydrotest'],
        'Col_12': df_in_long_line['Temperature Hydrotest'],
        'Col_13': df_in_long_line['Pressure Operation'],
        'Col_14': df_in_long_line['Temperature Operation']
        })
    df_out_second_line_1.set_index([pd.Series(df_in_long_line['index_copy'])], inplace = True)

    # Create a dataframe for the second rows of the *ROUTE section
    df_out_second_line_2 = pd.DataFrame({
        'Col_1': df_in_short_line['RLT'],
        'Col_2': df_in_short_line['Pressure Installation'],
        'Col_3': df_in_short_line['Temperature Installation'],
        'Col_4': df_in_short_line['Pressure Hydrotest'],
        'Col_5': df_in_short_line['Temperature Hydrotest'],
        'Col_6': df_in_short_line['Pressure Operation'],
        'Col_7': df_in_short_line['Temperature Operation']
        })

    # Create a dataframe for the first line of the *ROUTE section and set index to a large int
    df_out_last_line = pd.DataFrame({
        'Col_1': ['END'],
        'Col_2': ['0'],
        'Col_3': ['1'],
        'Col_4': ['2'],
        'Col_5': [f"{df_in['Friction Set'].iloc[-1]:.0f}"],
        'Col_6': [f"{df_in['Pipe Set'].iloc[-1]:.0f}"],
        'Col_7': ['0'],
        'Col_8': [df_in['RLT'].iloc[-1]],
        'Col_9': [df_in['Pressure Installation'].iloc[-1]],
        'Col_10': [df_in['Temperature Installation'].iloc[-1]],
        'Col_11': [df_in['Pressure Hydrotest'].iloc[-1]],
        'Col_12': [df_in['Temperature Hydrotest'].iloc[-1]],
        'Col_13': [df_in['Pressure Operation'].iloc[-1]],
        'Col_14': [df_in['Temperature Operation'].iloc[-1]]
        })
    df_out_last_line.index = [99999]

    # Create a dataframe for all the rows of the *ROUTE section
    df_out_temp = pd.concat([df_out_title, df_out_first_line, df_out_second_line_1,
                             df_out_second_line_2, df_out_last_line])
    df_out_temp = df_out_temp.sort_index()

    # Add the *ROUTE section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)

    return df_in_long_line_copy, df_out

def func_type(df_in, df_in_end, df_out):

    '''
    Add the *TYPE section to the Buckfast input file.

    Parameters
    ----------
    df_in : Dataframe
        Dataframe containing data from the BuckPy input file.
    df_in_end : Dataframe
        Dataframe containing data from the route ends of the BuckPy input file.
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Calculate property set
    # df_in['Route Type'] = df_in['Route Type'].str.capitalize() #! Use UPPER case always
    df_in_second_line = df_in

    # Calculate cbf mean, cbf std, residual force and residual length
    df_in_second_line[['mean_cbf_hydr', 'std_cbf_hydr', 'mean_cbf_oper', 'std_cbf_oper']] = \
        df_in_second_line.apply(lambda row: pd.Series(calc_cbf_mean_std(
            row['Route Type'], row['OD'], row['WT'], row['E'], row['Bend Radius'],
            row['Sleeper Height'], row['RCM Buckling Force'], row['sw Hydrotest'],
            row['sw Operation'], row['Lateral Hydrotest Mean'], row['Lateral Hydrotest STD'],
            row['Lateral Operation Mean'], row['Lateral Operation STD'], row['HOOS Mean'],
            row['HOOS STD'])), axis = 1)

    # Replace "RCM" in "Route Type" column to "SLEEPER"
    df_in_second_line['Route Type'] = df_in_second_line['Route Type'].replace("RCM", "SLEEPER")

    # Create a dataframe for the title of the *TYPE section
    df_out_title = pd.DataFrame({
        'Col_1': ['*TYPE']
        })

    # Create a dataframe for the first line of the *TYPE section
    df_out_first_line = pd.DataFrame({
        'Col_1': ['END'],
        'Col_2': ['1'],
        'Col_3': df_in_end['Route Type'].iloc[0],
        'Col_4': f"{-1*df_in_end['Reaction Installation'].iloc[0]:.0f}",
        'Col_5': f"{-1*df_in_end['Reaction Hydrotest'].iloc[0]:.0f}",
        'Col_6': f"{-1*df_in_end['Reaction Operation'].iloc[0]:.0f}"
        })

    # Create a dataframe for the rows in second line of the *TYPE section
    df_out_second_line = pd.DataFrame({
        'Col_1': df_in_second_line['Route Type'].str.upper(),
        'Col_2': df_in_second_line['Property Set'].astype(int).astype(str),
        'Col_3': df_in_second_line['cbf_no'].astype(int).astype(str),
        'Col_4': CBF_DIST,
        'Col_5': -1*df_in_second_line['mean_cbf_hydr'],
        'Col_6': df_in_second_line['std_cbf_hydr'],
        'Col_7': df_in_second_line['HOOS Reference Length'].astype(int).astype(str),
        'Col_8': (-1*df_in_second_line['Residual Buckle Force Hydrotest']
                  ).astype(int).astype(str),
        'Col_9': df_in_second_line['Residual Buckle Length Hydrotest'].astype(int).astype(str),
        'Col_10': CBF_DIST,
        'Col_11': -1*df_in_second_line['mean_cbf_oper'],
        'Col_12': df_in_second_line['std_cbf_oper'],
        'Col_13': df_in_second_line['HOOS Reference Length'].astype(int).astype(str),
        'Col_14': (-1*df_in_second_line['Residual Buckle Force Operation']
                   ).astype(int).astype(str),
        'Col_15': df_in_second_line['Residual Buckle Length Operation'].astype(int).astype(str)
        })

    # Create a dataframe for the last line of the *TYPE section
    df_out_third_line = pd.DataFrame({
        'Col_1': ['END'],
        'Col_2': ['2'],
        'Col_3': df_in_end['Route Type'].iloc[-1],
        'Col_4': f"{-1*df_in_end['Reaction Installation'].iloc[-1]:.0f}",
        'Col_5': f"{-1*df_in_end['Reaction Hydrotest'].iloc[-1]:.0f}",
        'Col_6': f"{-1*df_in_end['Reaction Operation'].iloc[-1]:.0f}"
        })
    df_out_third_line.index = [99999]

    # Create a dataframe for all the rows of the *TYPE section
    df_out_temp = pd.concat([df_out_title, df_out_first_line,
                             df_out_second_line, df_out_third_line])

    # Add the *TYPE section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)

    return df_out

def func_output(df_out):

    '''
    Add the *OUTPUT section to the Buckfast input file.

    Parameters
    ----------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Create a dataframe for the title of the *OUTPUT section
    df_out_title = pd.DataFrame({
        'Col_1': ['*OUTPUT']
        })

    # Create a dataframe for the first line of the *OUTPUT section
    if DEFAULT_OUTPUT:
        df_out_first_line = pd.DataFrame({
            'Col_1': ['0.01'],
            'Col_2': ['0.10'],
            'Col_3': ['200'],
            'Col_4': ['0'],
            'Col_5': ['10000'],
            'Col_6': ['-1'],
            'Col_7': ['0'],
            'Col_8': ['0'],
            'Col_9': ['1'],
            'Col_10': ['-1.0E+07'],
            'Col_11': ['0'],
            })
    else: # Self defined values of the *OUTPUT section
        df_out_first_line = pd.DataFrame({
            'Col_1': ['0.01'],
            'Col_2': ['0.10'],
            'Col_3': ['200'],
            'Col_4': ['0'],
            'Col_5': ['10000'],
            'Col_6': ['-10'],
            'Col_7': ['0'],
            'Col_8': ['0'],
            'Col_9': ['10'],
            'Col_10': ['-4.0E+06'],
            'Col_11': ['0'],
            })

    # Create a dataframe for all the rows of the *OUTPUT section
    df_out_temp = pd.concat([df_out_title, df_out_first_line])

    # Add the *OUTPUT section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)
    return df_out

def func_end(df_out):

    '''
    Add the *END section to the Buckfast input file.

    Parameters
    ----------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.

    Returns
    -------
    df_out : Dataframe
        Dataframe containing the Buckfast input file.
    '''

    # Create a dataframe for the row of the *END section
    df_out_temp = pd.DataFrame({
        'Col_1': ['*END']
        })

    # Add the *END section to the Buckfast input file
    df_out = pd.concat([df_out, df_out_temp], ignore_index = True)
    return df_out

#################################################
# Main
#################################################

WORK_DIR = r'C:\Github_repos\BuckPy\source'   # The BuckPy working directory
BUCKPY_EXCEL = 'inputDataTemplateA.xlsx'      # The BuckPy Excel template file
DESCRIPTION = 'PBLA'                          # Description of the BuckPy project
PIPELINE_LIST = ['A']                         # Pipeline IDs, ['A', 'B']
SCENARIO_LISTS = [[1]]                        # Scenario numbers for each pipeline, [[1,2], [1,2]]

# Switch to use default values of *OUTPUT section
DEFAULT_OUTPUT = True # True or False

# Define essential parameters
WATER_DENSITY = 1025           # Unit: kg/m^3
AXIAL_FRIC_DIST = 'LOGNORMAL'  # Distribution type for axial friction, NORMAL or LOGNORMAL
COAT_WT = 0.3                  # Thickness of the external coating, unit: mm
STEEL_DENSITY = 7850           # Density of steel, unit: kg/m^3
COAT_LAYER = 1                 # Number of layers of external coating
CBF_NO = 2                     # Number of critical buckling force distribution, 1 or 2
CBF_DIST = 'LOGNORMAL'         # Distribution type for critical buckling force, NORMAL or LOGNORMAL

for index, PIPELINE_ID in enumerate(PIPELINE_LIST):

    SCENARIO_LIST = SCENARIO_LISTS[index]
    for SCENARIO_NO in SCENARIO_LIST:

        OUTPUT_CSV_NAME = f'buckfast_{PIPELINE_ID}_scen{SCENARIO_NO}.csv'
        print(f'Writing Buckfast input file: {OUTPUT_CSV_NAME}')

        # Read the BuckPy input file into a dataframe
        df_input, df_input_ends = open_file()

        # Add the *HEADING section to the Buckfast input file
        df_output = func_heading(df_input)

        # Add the *SIMULATIONS section to the Buckfast input file
        df_output = func_simulations(df_input, df_output)

        # Add the *WATER section to the Buckfast input file
        df_output = func_water(df_input, df_output)

        # Add the *AXIALFRICTION section to the Buckfast input file
        df_output = func_friction(df_input, df_output)

        # Add the *SINGLE section to the Buckfast input file
        df_output = func_single(df_input, df_output)

        # Add the *INSTALLATION section to the Buckfast input file
        df_output = func_installation(df_input, df_output)

        # Add the *HYDROTEST section to the Buckfast input file
        df_output = func_hydrotest(df_input, df_output)

        # Add the *OPERATION section to the Buckfast input file
        df_output = func_operation(df_input, df_output)

        # Add the *ROUTE section to the Buckfast input file
        df_input_type, df_output = func_route(df_input, df_output)

        # Add the *TYPE section to the Buckfast input file
        df_output = func_type(df_input_type, df_input_ends, df_output)

        # Add the *OUTPUT section to the Buckfast input file
        df_output = func_output(df_output)

        # Add the *END section to the Buckfast input file
        df_output = func_end(df_output)

        # Write Buckfast input file
        df_output.to_csv(rf'{WORK_DIR}\{OUTPUT_CSV_NAME}', index = False, header = False)
