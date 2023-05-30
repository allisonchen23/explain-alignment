import numpy as np
import pandas as pd
import os, sys

def string_to_numpy(string, verbose=False):
    '''
    Given a string, convert it to a numpy array

    Arg(s):
        string : str
            string assumed in format of numbers separated by spaces, with '[' ']' on each end
        verbose : bool
            whether or not to print error messages
    Returns:
        np.array
    '''
    if type(string) != str:
        return string
    original_string = string

    if string[0] == '[':
        string = string[1:]
    if string[-1] == ']':
        string = string[:-1]

    string = string.split()
    try:
        string = [eval(i) for i in string]
    except:
        if verbose:
            print("Unable to convert '{}' to numbers. Returning original string".format(original_string))
        return original_string

    return np.array(string)

def convert_string_columns(df, columns=None):
    '''
    Given a dataframe, convert columns to numpy if they are strings

    Arg(s):
        df : pd.DataFrame
            Original dataframe
        columns : list[str] or None
            If None, iterate all the columns

    Returns:
        df : modified dataframe with strings replaced with numpy.array
    '''

    if columns == None:
        columns = df.columns

    for column in columns:
        df[column] = df[column].map(string_to_numpy)
    return df