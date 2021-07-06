import os
import numpy as np
import pandas as pd
from fileutils import *

'''
Setting up paths relevant to clinical data and serialized
clinical data
'''
T1_DTI_path = os.path.join(raw_data_path, 'T1_DTI')
combat_path = os.path.join(raw_data_path, 'combat')


def clean_DTI(overwrite=False):
           
    DTI_cleaned_path = os.path.join(data_path, 'DTI', 'DTI.csv')
    
    if not overwrite and os.path.exists(DTI_cleaned_path):
        return
        
    # Renaming and dropping columns
    DTI_path = os.path.join(T1_DTI_path, 'DTI_cleaned_v3.csv')
    df = pd.read_csv(DTI_path)
    
    rename_dict = {}
    
    cols = df.columns.tolist()
    
    for col in cols:
        if 'Unnamed' in col:
            df.drop(labels=[col], axis=1, inplace=True)
    cols = df.columns.tolist()
    
    for c in cols:
        if c == 'SubjID':
            rename_dict[c] = 'SubjectID'
        elif 'CurrPTSDdx' in c:
            rename_dict[c] = 'Diagnosis'
        elif 'Sex' in c:
            rename_dict[c] = 'Sex'
            
    df.rename(columns=rename_dict, inplace=True)
    
    df.drop(columns=['SiteID'], inplace=True)
    
    df = df[~((df == 'NA').any(1))]
    df.dropna(subset=['Age', 'Sex', 'Diagnosis'], inplace=True)
    
    # Remapping values to match other datasets
    ##diag_map = {'TEHC':'Control', 'HC':'Control' 'PTSD':'PTSD'}
    sex_map = {'Male':'M', 'Female':'F'}
    
    ##df['Diagnosis'] = df['Diagnosis'].map(diag_map)
    df['Sex'] = df['Sex'].map(sex_map)
    
    df.to_csv(DTI_cleaned_path, index=False)
    

def load_DTI_cleaned():
    '''
    Load the cleaned DTI file
    
    return:
        DataFrame containing DTI data for all subjects from all sites
    '''
    csv_path = os.path.abspath(os.path.join(data_path, 'DTI', 'DTI.csv'))
    final_df = pd.read_csv(csv_path)
    return final_df
    

def clean_T1(overwrite=False):
        
    T1_cleaned_path = os.path.join(data_path, 'T1', 'T1.csv')
    
    if not overwrite and os.path.exists(T1_cleaned_path):
        return
    
    # Renaming and dropping columns
    T1_path = os.path.join(T1_DTI_path, 'T1_cleaned_v2.csv')
    df = pd.read_csv(T1_path)
    
    rename_dict = {}
    
    cols = df.columns.tolist()
    
    for col in cols:
        if 'Unnamed' in col:
            df.drop(labels=[col], axis=1, inplace=True)
    cols = df.columns.tolist()
    
    for c in cols:
        if c == 'SubjID':
            rename_dict[c] = 'SubjectID'
        elif 'CurrPTSDdx' in c:
            rename_dict[c] = 'Diagnosis'
            
    df.rename(columns=rename_dict, inplace=True)
    
    df.dropna(subset=['Age', 'Sex', 'Diagnosis'], inplace=True)
    
    # Remapping values to match other datasets
    ##diag_map = {'TEHC':'Control', 'HC':'Control', 'SubThresh':'Subthreshold', 'PTSD':'PTSD', 'Control':'Control'}
    sex_map = {'Male':'M', 'Female':'F'}
    
    ##df['Diagnosis'] = df['Diagnosis'].map(diag_map)
    df['Sex'] = df['Sex'].map(sex_map)
    
    df.to_csv(T1_cleaned_path, index=False)
    
    
def load_T1_cleaned():
    '''
    Load the cleaned T1 file
    
    return:
        DataFrame containing T1 data for all subjects from all sites
    '''
    csv_path = os.path.abspath(os.path.join(data_path, 'T1', 'T1.csv'))
    final_df = pd.read_csv(csv_path)
    return final_df
    

def clean_T1_combat(overwrite=False):
        
    T1_cleaned_path = os.path.join(data_path, 'combat', 'T1', 'T1_combat.csv')
    
    if not overwrite and os.path.exists(T1_cleaned_path):
        return
    
    # Renaming and dropping columns
    T1_path = os.path.join(combat_path, 'T1', 'T1_combat.csv')
    df = pd.read_csv(T1_path)
    
    df.replace("", np.nan, inplace=True)
    df.replace("NA", np.nan, inplace=True)
    site_diag_filt = df.columns.isin(["Site", "Diagnosis"])
    df[df.columns[~site_diag_filt]] = df[df.columns[~site_diag_filt]].astype(np.float64)
    
    df.to_csv(T1_cleaned_path, index=False)
    
    
def load_T1_combat_cleaned():
    '''
    Load the cleaned T1 file
    
    return:
        DataFrame containing T1 data for all subjects from all sites
    '''
    csv_path = os.path.abspath(os.path.join(data_path, 'combat', 'T1', 'T1_combat.csv'))
    final_df = pd.read_csv(csv_path)
    return final_df


if __name__ == "__main__":
    ##clean_DTI(overwrite=True)
    clean_T1(overwrite=True)
    ##clean_T1_combat(overwrite=True)
    
    df = load_T1_combat_cleaned()
    print(len(df.index))
    print(df.columns.tolist())
    