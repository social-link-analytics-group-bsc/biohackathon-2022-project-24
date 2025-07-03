'''
USAGE
From python script (dict):
    from get_cleaned_data import get_cleaned_data
    dict = get_cleaned_data(dict)

From python script (list of dicts):
    from get_cleaned_data import get_cleaned_data
    dict = get_cleaned_data(dict, input_type = 'list_of_dicts')
    
From bash terminal (json file (dict or list of dicts))
    python get_cleaned_data.py <json_input> <json_output>

'''
import pandas as pd
import ast
import json
import sys

def list_to_num(item):
    '''
    Convert lists from each sentence to a number:
    - len(list) == 0 --> None
    - len(list) == 1 --> list[0]
    - len(list) >= 2 --> max(list)
    '''
    if item is None:
        number = None
    elif len(item) == 0:
        number = None
    elif len(item) == 1:
        number = item[0]
    else:
        # Get the maximum value
        # TODO - maybe change
        number = max(item)
    return number

def PMCID_to_sentences(df, relevant_columns):
    '''Explode each PMCID into its sentences'''

    # Convert str() to list() with ast library
    # Convert 'null' to None
    for colname in relevant_columns:
        df[colname] = df[colname].map(lambda x: ast.literal_eval(x.replace('null', 'None')))

    # Check if any row has information. If it doesn't, return a DataFrame of None values
    if all(df['sentence_index'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)):
        sys.stderr.write(f'WARNING: PMCID {df["pmcid"].tolist()} Has no sentences with data.')
        for colname in relevant_columns:
            df[colname] = df[colname].map(list_to_num)
        return df

    # Apply zip to align the lists, then explode
    df['zipped'] = [list(zip(*[row[colname] for colname in relevant_columns])) for index, row in df.iterrows()]

    # Explode lists to several columns
    exploded_df = df.explode('zipped')

    # Create separate columns from the zipped tuples
    exploded_df[relevant_columns] = pd.DataFrame(exploded_df['zipped'].tolist(), index=exploded_df.index)

    # Drop the temporary zipped column
    exploded_df = exploded_df.drop('zipped', axis=1)

    # Reset index if needed
    exploded_df = exploded_df.reset_index(drop=True)

    # Convert lists to numbers
    for colname in relevant_columns:
        if colname != 'sentence_index':
            exploded_df[colname] = exploded_df[colname].map(list_to_num)

    return exploded_df


symbol_to_col = {'F': 'n_fem', 'M': 'n_male',
                'PF': 'perc_fem', 'PM': 'perc_male',
                'S': 'sample'}

def is_na(row, symbol):
    cell = row[symbol_to_col[symbol]]
    # If cell is a list or array, check if any or all elements are NaN or None
    if isinstance(cell, list):
        return any(pd.isna(x) or x is None for x in cell)
    # For single values
    return pd.isna(cell) or cell is None

def not_na(row, symbol):
    # The opposite of is_na
    return not is_na(row, symbol)

def categorize_row(row):
    # Full row
    if not_na(row, 'F') & not_na(row, 'M') & not_na(row, 'PF') & not_na(row, 'PM') & not_na(row, 'S'):
        return 'Full row'

    # F & M information
    elif not_na(row, 'F') & not_na(row, 'M'):
        if not_na(row, 'S'):
            if is_na(row, 'PF') & is_na(row, 'PM'):
                return 'F&M&S'
            else:
                return 'F&M&S & PF|PM'
        elif not_na(row, 'PF') & not_na(row, 'PM'):
            return 'F&M & PF|PM'
        else:
            return 'F&M'

    # F information
    elif not_na(row, 'F'):
        if not_na(row, 'S'):
            if is_na(row, 'PF') & is_na(row, 'PM'):
                return 'F&S'
            else:
                return('F&S & PF|PM')
        else:
            if is_na(row, 'PF') & is_na(row, 'PM'):
                return 'F'
            else:
                if not_na(row, 'PF'):
                    return('F&PF')
                else:
                    return('F&PM')

    # M information
    elif not_na(row, 'M'):
        if not_na(row, 'S'):
            if is_na(row, 'PF') & is_na(row, 'PM'):
                return 'M&S'
            else:
                return('M&S & PF|PM')
        else:
            if is_na(row, 'PF') & is_na(row, 'PM'):
                return 'M'
            else:
                if not_na(row, 'PF'):
                    return('M&PF')
                else:
                    return('M&PM')

    # Information on percentage M & F
    elif not_na(row, 'PF') & not_na(row, 'PM'):
        if not_na(row, 'S'):
            return 'PF&PM&S'
        else:
            return 'PF&PM'

    # Information only on percentage of M|F
    elif not_na(row, 'PF'):
        if not_na(row, 'S'):
            return 'PF&S'
        else:
            return 'PF'
    elif not_na(row, 'PM'):
        if not_na(row, 'S'):
            return 'PM&S'
        else:
            return 'PM'
    elif not_na(row, 'S'):
        return 'S'
    elif is_na(row, 'F') & is_na(row, 'M') & is_na(row, 'PF') & is_na(row, 'PM') & is_na(row, 'S'):
        return 'Empty row'
    else:
        return 'Other'

def assign_categories(df, relevant_columns):
    '''
    Assign one category to each sentence
    Output: df with new column:
        category_type
    '''
    
    # Apply categorize_row function to each row of the DataFrame
    df['category_type'] = df.apply(categorize_row, axis=1)

    # Count the number of occurrences for each category
    category_counts = df['category_type'].value_counts()

    print('CATEGORY COUNTS')
    print(f'Total\t      {len(df)}')
    print(category_counts)

    return df


def populate_sentences(df):
    '''
    Populate sentences given a set of rules.
    If the sentence doesn't pertain to any rule, leave as NaN
    Output: df with 5 new columns:
        clean_sample, clean_n_fem, clean_n_male, clean_perc_fem, clean_perc_male
    '''
    # Add new columns
    new_cols = ['clean_n_fem', 'clean_n_male', 'clean_perc_fem',	'clean_perc_male',	'clean_sample']
    for column in new_cols:
        df[column] = None


    ## Categories ['F&M', 'F&M&S', 'F&M&S & PF|PM', 'F&M & PF|PM', 'Full row'])
    ## Use F&M info to infer the rest
    # TODO - check in how many cases S != M+F.
    #        Also, maybe I'm adding too many things and should just drop them instead
    mask = df['category_type'].isin(['F&M', 'F&M&S', 'F&M&S & PF|PM', 'F&M & PF|PM', 'Full row'])

    F_df =  df[mask]['n_fem']
    M_df = df[mask]['n_male']

    df.loc[mask, 'clean_n_fem'] =     F_df
    df.loc[mask, 'clean_n_male'] =    M_df
    df.loc[mask, 'clean_sample'] =    F_df + M_df
    df.loc[mask, 'clean_perc_fem'] =  round(F_df/(F_df + M_df)*100, 3)
    df.loc[mask, 'clean_perc_male'] = round(M_df/(F_df + M_df)*100, 3)


    ## Categories ['F&S', 'F&S & PF|PM'], ['M&S', 'M&S & PF|PM']
    ## Use S & F|M info to ifer the rest
    for sex1, sex2 in [['fem', 'male'], ['male', 'fem']]:
        # Discard rows where sample < male/female values
        if sex1 == 'fem':
            mask = df['category_type'].isin(['F&S', 'F&S & PF|PM'])
            if mask.any():
                mask &= (df['sample'] > df['n_fem'])
        else:
            mask = df['category_type'].isin(['M&S', 'M&S & PF|PM'])
            if mask.any():
                mask &= (df['sample'] > df['n_fem'])
        
        S_df = df[mask]['sample']
        sex1_df = df[mask][f'n_{sex1}']

        df.loc[mask, f'clean_n_{sex1}'] =     sex1_df
        df.loc[mask, f'clean_n_{sex2}'] =     S_df - sex1_df
        df.loc[mask, 'clean_sample'] =        S_df
        df.loc[mask, f'clean_perc_{sex1}'] =  round(sex1_df/S_df*100, 3)
        df.loc[mask, f'clean_perc_{sex2}'] =  round((S_df - sex1_df)/S_df*100, 3)

    
    ## Categories ['PF&S', 'PM&S']
    ## Use S & PF|PM info to infer the rest
    for sex1, sex2 in [['fem', 'male'], ['male', 'fem']]:
        if sex1 == 'fem':
            mask = df['category_type'] == 'PF&S'
        else:
            mask = df['category_type'] == 'PM&S'
        S_df = df[mask]['sample']
        sex1_perc_df = df[mask][f'perc_{sex1}']

        df.loc[mask, f'clean_n_{sex1}'] =     round(S_df*sex1_perc_df/100)
        df.loc[mask, f'clean_n_{sex2}'] =     round(S_df*(100 - sex1_perc_df)/100)
        df.loc[mask, 'clean_sample'] =        S_df
        df.loc[mask, f'clean_perc_{sex1}'] =  sex1_perc_df
        df.loc[mask, f'clean_perc_{sex2}'] =  100 - sex1_perc_df

    
    ## Category ['PF&PM&S']
    ## Use PF, PM & S to infer F and M
    # Discard if PF+PM != 100
    mask = df['category_type'] == 'PF&PM&S'
    # Discard if percentages don't add up to 100
    mask &= (round(df[mask]['perc_fem'] + df[mask]['perc_male']) == 100)
    S_df = df[mask]['sample']
    PF_df = df[mask][f'perc_fem']
    PM_df = df[mask][f'perc_male']

    df.loc[mask, 'clean_n_fem'] =     round(S_df * PF_df / 100)
    df.loc[mask, 'clean_n_male'] =    round(S_df * PM_df / 100)
    df.loc[mask, 'clean_sample'] =    S_df
    df.loc[mask, 'clean_perc_fem'] =  PF_df
    df.loc[mask, 'clean_perc_male'] = PM_df


    ## Categories ['F&PF', 'M&PM']
    ## Use F&PF | M&PM info to infer the rest
    for sex1, sex2 in [['fem', 'male'], ['male', 'fem']]:
        if sex1 == 'fem':
            mask = df['category_type'] == 'F&PF'
        else:
            mask = df['category_type'] == 'M&PM'
        sex1_df = df[mask][f'n_{sex1}']
        sex1_perc_df = df[mask][f'perc_{sex1}']

        df.loc[mask, f'clean_n_{sex1}'] =     sex1_df
        df.loc[mask, f'clean_n_{sex2}'] =     round(sex1_df*100/sex1_perc_df) - sex1_df
        df.loc[mask, 'clean_sample'] =        round(sex1_df*100/sex1_perc_df)
        df.loc[mask, f'clean_perc_{sex1}'] =  sex1_perc_df
        df.loc[mask, f'clean_perc_{sex2}'] =  100 - sex1_perc_df


    ## Categories ['F', 'M']
    ## If all sentences with the PMCID are from the category F|M, then populate, considering 
    ## there are no samples of the other sex (e.g. if F=number, then M=0)
    for sex1, sex2 in [['fem', 'male'], ['male', 'fem']]:

        # Find pmcids that are only associated with 'F'/'S' category
        if sex1 == 'fem':
            pmcids_with_only_MF = df.groupby('pmcid').filter(lambda x: (x['category_type'] == 'F').all()).pmcid.unique()
            # onlyMF_df = df[df["Category"] == 'F']
        else:
            pmcids_with_only_MF = df.groupby('pmcid').filter(lambda x: (x['category_type'] == 'M').all()).pmcid.unique()
            # onlyMF_df = df[df["Category"] == 'M']

        # Filter out rows from df where pmcid is in pmcids_with_only_S
        mask = df['pmcid'].isin(pmcids_with_only_MF)
        sex1_df = df[mask][f'n_{sex1}']

        # Populate dataset
        df.loc[mask, f'clean_n_{sex1}'] =     sex1_df
        df.loc[mask, f'clean_n_{sex2}'] =     0
        df.loc[mask, 'clean_sample'] =        sex1_df
        df.loc[mask, f'clean_perc_{sex1}'] =  100
        df.loc[mask, f'clean_perc_{sex2}'] =  0


    ## Category ['S']
    ## Only fill in 'clean_sample' if that PMCID only has sentences with 'S' category

    # Identify pmcids that have at least one non-'S' category
    pmcids_with_non_S = df[df['category_type'] != 'S']['pmcid'].unique()
    # Create a mask for rows where category is 'S' but pmcid is not in pmcids_with_non_S
    mask = (df['category_type'] == 'S') & (~df['pmcid'].isin(pmcids_with_non_S))
    # Fill in 'clean_sample' dataset
    sample_df = df[mask]['sample']
    df.loc[mask, 'clean_sample'] = sample_df


    ## Categories 'M&PF', 'F&PM', 'Empty row'
    ## Leave empty

    ## 'category_used' column is True for filled in columns, False otherwise
    df['category_used'] = (df['clean_sample'].notna()) & (df['clean_sample'] != 0)

    return df


def merge_sentences(df):
    '''
    Merge sentences by calculating:
    - Sum of clean_sample, clean_n_fem, clean_n_male
    - Mean of clean_perc_fem, clean_perc_fem
    Output: df with 1 row [per PMCID] and changed columns:
    - category_type: list of categories by sentence
    - category_use: boolean list of whether sentences were populated
    '''

    # Group by 'pmcid' and aggregate the data
    df = df.groupby('pmcid').agg({
        'clean_n_fem': 'sum',
        'clean_n_male': 'sum',
        'clean_perc_fem': 'mean',
        'clean_perc_male': 'mean',
        'clean_sample': 'sum',
        'category_type': list,
        'category_used': list,
        'sample': list,
        'sentence_index': lambda x: [int(val) if not pd.isna(val) else None for val in x]
    }).reset_index()

    # Convert from float to integers some rows
    df['clean_n_fem'] =  df['clean_n_fem'].astype(int)
    df['clean_n_male'] = df['clean_n_male'].astype(int)
    df['clean_sample'] = df['clean_sample'].astype(int)

    # Fix the value for PMCIDs with only S info to be max() instead of sum()
    # Get the max value for the list in column 'sample'
    mask = df['clean_perc_fem'].isna() & (df['clean_sample'] > 0)
    if mask.any():
        df.loc[mask, 'clean_sample'] = df.loc[mask, 'sample'].apply(lambda x: max(x) if isinstance(x, list) else x)

    return df

def get_clean_data(input_dict, input_type = 'dict'):
    '''
    Input: dict, or list of dicts, with at least the following keys, where all values are strings:
         ['pmcid', 'sentence_index', 'n_fem', 'n_male', 'perc_fem', 'perc_male', 'sample']

    Output: list of dict/s with keys: 
        'pmcid'
             (str)
        'clean_n_fem', 'clean_n_male', 'clean_sample'
             (int / None)
        'clean_perc_fem', 'clean_perc_male'
            (float)
        'category_type', 'category_used', 'sentence_index'
            (list)

    '''
    relevant_columns = ['sentence_index', 'n_fem', 'n_male', 'perc_fem', 'perc_male', 'sample']
    
    # Convert to pandas, checking if input is a dict, or list of dicts
    if input_type == 'dict':
        if isinstance(input_dict, dict):
            df = pd.DataFrame([input_dict])
        else:
            raise ValueError('Input is not dict')
    elif input_type == 'list_of_dicts':
        if isinstance(input_dict, list) & isinstance(input_dict[0], dict):
            df = pd.DataFrame(input_dict)
        else:
            raise ValueError('Input is not a list of dicts')
    else:
        raise ValueError('input type should be "dict" or "list_of_dicts')

    # TODO - remove
    #df = df[['pmcid'] + relevant_columns][df['pmcid'] == 'PMC9278617']

    # Explode into sentences
    df = PMCID_to_sentences(df, relevant_columns)

    # Assign a category to each sentence
    df = assign_categories(df, relevant_columns)

    # Use rules to populate some sentences
    df = populate_sentences(df)
    # Merge the sentences back to 1 per PMCID
    df = merge_sentences(df)
    # Return dictionary
    final_df = df.drop(['sample'], axis=1)

    output_list = final_df.to_dict(orient='records')

    if input_type == 'dict':
        output_dict = output_list[0]
        return output_dict

    return output_list


if __name__ == '__main__':
    '''
    Input a json file and output a JSON file.
    JSON file can be either a dict or a list of dicts
    '''

    # Get a JSON file
    if len(sys.argv) != 3:
        raise IndexError("Usage: python script.py <input> <output>")
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load dictionary
    with open(input_path) as file:
        input_dict = json.load(file)

    # Get the cleaned data
    if isinstance(input_dict, dict):
        output_dict = get_clean_data(input_dict)
    elif isinstance(input_dict, list) & isinstance(input_dict[0], dict):
        output_dict = get_clean_data(input_dict, input_type = 'list_of_dicts')
    else:
        raise ValueError('Input is not a dict nor a list of dicts')

    # Save dictionary
    with open(output_path, 'w') as file:
        json.dump(output_dict, file, indent=4)
    print(f"JSON data has been saved to {output_path}")




