import os
import numpy as np
import pandas as pd
from atlas_to_file import *
from fileutils import *


'''
Setting up paths relevant to clinical data and serialized
clinical data
'''
clinical_path = os.path.join(raw_data_path, "clinical_data")
feather_path = os.path.join(serialized_path, 'raw_data', 'clinical')


def serialize_clinical(overwrite=False):
    '''
    Serialize all the original clincal data files
    
    params:
        overwrite : bool (default = False)
            Whether to overwrite the existing serialized data
    '''
    # Generate a map for mapping different csv headers to a unified header
    sheet_map_raw = {('default',):('Clinical data',), ('Munster', 'NanjingYixing','Tours','Utrecht','Vanderbilt', 'WesternOntario', 'Westhaven_VA', 'Wisc_Cisler', 'Wisc_Grupe'):('Clinicaldemographic data',), ('Stanford',):('Clinicaldemographic data Brains', 'Clinicalsemographic data CC'), ('Toledo',):('demographics',), ('McLean', 'Michigan',):('Clinical Data',), ('Waco_VA',):('ENIGMA_Subject_Info',)}
    
    sheet_map = {}
    for k, v in sheet_map_raw.items():
        for e in k:
            sheet_map[e] = v
    
    # Filter files by site name
    files = os.listdir(clinical_path)
    unique_sites = load_site_subject()['Site'].unique()

    files_filtered = []
    for file in files:
        filename, ext = os.path.splitext(file)
        if filename in unique_sites:
            files_filtered.append(file)
    
    # If the serialized folder path doesn't exist, make it
    if not os.path.exists(feather_path):
        os.makedirs(feather_path)
    
    # Iterate over all of the clinical data excel files and serialize them
    for file in files_filtered:
        filename, ext = os.path.splitext(file)
        
        feather_name = os.path.join(feather_path, '{}.ftr'.format(filename))
        
        # If we are not overwriting and the path already exists, go to next file
        if not overwrite and os.path.exists(feather_name):
            continue
        
        if filename not in sheet_map:
            sheet_map[filename] = sheet_map['default']
            
        sheets = sheet_map[filename]
        
        df_final = None
        
        for sheet in sheets:
            if sheet is None:
                try:
                    df = pd.read_excel(os.path.join(clinical_path, file))
                except Exception as e:
                    print(file)
                    raise e
            else:
                try:
                    df = pd.read_excel(os.path.join(clinical_path, file), sheet_name=sheet)
                except Exception as e:
                    print(file)
                    raise e
                    
            if df_final is None:
                df_final = df
            else:
                df_final = pd.concat([df_final, df], ignore_index=True, join="inner")
                
        for column, dt in zip(df_final.dtypes.index, df_final.dtypes):
            if dt == 'object':
                df_final[column] = df_final[column].astype(str)
        
        try:
            df_final.to_feather(feather_name)
        except Exception as e:
            print(file)
            raise e
                

def clean_clinical(overwrite=False):
    '''
    Cleaning up the clinical data so that it is consistent and in a single file
    
    params:
        overwrite: bool (default = False)
            Whether to overwrite the existing clinical data file
    '''
    serialize_clinical(overwrite=False)
    
    # ID column names
    # Keeping out 'ID' because many have both ID and one of id_options
    id_options = ['Resting State ID', 'Scan IDs', 'Scan ID']
    # PTSD diagnosis column names
    diagnosis_options = ['Current PTSD diagnosis', 'CurrPTSDdx']
    
    '''
    diagnosis_elements = {'trauma-exposed adults without PTSD':'TEHC', 'PTSD':'PTSD', 'Trauma Control':'TEHC', 'Healthy control':'HC', 'Control':'Control', 'Subthreshold':'Subthreshold'}
    '''
    diagnosis_elements = {'trauma-exposed adults without PTSD':'Control', 'PTSD':'PTSD', 'Trauma Control':'Control', 'Healthy control':'Control', 'Control':'Control', 'Subthreshold':'Subthreshold'}
    
    clinical_new_path = os.path.join(data_path, 'clinical_data')
    
    site_subject = load_site_subject()
    site_subject['SubjectID'] = site_subject['SubjectID'].astype(str)
    
    clinical_df_final = pd.DataFrame()
    
    col_list = []
    
    for f in os.listdir(feather_path):
        print(f)
        fpath = os.path.join(feather_path, f)
        fname, _ = os.path.splitext(f)
        clinical_df = pd.read_feather(fpath)
        ##print(clinical_df.columns)
        col_list.append(clinical_df.columns)
        clinical_df_clean = pd.DataFrame()
        
        # Cleaning ID column names and setting column
        if fname == 'Ghent':
            clinical_df_clean['SubjectID'] = clinical_df['ID']
        else:
            changed = False
            for id_opt in id_options:
                if id_opt in clinical_df.columns:
                    clinical_df_clean['SubjectID'] = clinical_df[id_opt]
                    changed = True
                    break
            if not changed:
                clinical_df_clean['SubjectID'] = clinical_df['ID']
            
        # Cleaning Diagnosis column names and setting column
        for diag_opt in diagnosis_options:
            if diag_opt in clinical_df.columns:
                clinical_df_clean['Diagnosis'] = clinical_df[diag_opt]
                break
                
        # Setting Age column
        clinical_df_clean['Age'] = clinical_df['Age']
        
        # Setting Sex column
        clinical_df_clean['Sex'] = clinical_df['Sex']
        
        # Mapping all IDs
        def map_id(s):
            if fname == 'Columbia':
                return str(int(s.replace('-', '')))
            elif fname == 'Leiden':
                return s.replace('Episca', '')
            elif fname == 'UMN':
                return 'MARS2_' + s;
            elif fname == 'UWash':
                return s.replace('R', '')
            elif fname == 'Wisc_Cisler':
                if 'DOP' in s or 'PAL' in s:
                    return s.replace(' ', '_')
                elif 'EMO' in s:
                    s_split = s.split('_')
                    return s_split[0] + s_split[1] + '_' + s_split[2]
                else:
                    return s
            else:
                return s
        
        clinical_df_clean['SubjectID'] = clinical_df_clean['SubjectID'].apply(map_id)
        
        # Rows 1 and 2 are messed up in UMN
        if fname == 'UMN':
            clinical_df_clean.drop([0,1], inplace=True)
        
        # Map all ID columns to string
        def id_to_string(x):
            if np.isnan(x):
                return ''
            else:
                return str(int(x))
            
        if clinical_df_clean['SubjectID'].dtype != 'object':
            clinical_df_clean['SubjectID'] = clinical_df_clean['SubjectID'].apply(id_to_string)
        
        # Mapping all Diagnosis
        def map_diagnosis(row):
            for diag_e, diag_map in diagnosis_elements.items():
                if diag_e.lower() in row['Diagnosis'].lower():
                    return diag_map
            return ''
        
        print(clinical_df_clean['Diagnosis'].unique())
        
        clinical_df_clean['Diagnosis'] = clinical_df_clean.apply(map_diagnosis, axis=1)
        
        # Mapping all Sex
        def map_sex(s):
            sex_map = {'M':'M', 'F':'F', 'Male':'M', 'Female':'F'}
            if s in sex_map:
                return sex_map[s]
            else:
                return ''
            
        clinical_df_clean['Sex'] = clinical_df_clean['Sex'].apply(map_sex)
        
        # Setting Site
        clinical_df_clean = clinical_df_clean.assign(Site = fname)
        
        # Reordering columns
        columns = ['Site', 'SubjectID', 'Diagnosis', 'Age', 'Sex']
        clinical_df_clean = clinical_df_clean[columns]
        
        # Sorting and filtering
        site_subject_curr = site_subject[site_subject['Site'] == fname]
        subject_filter = clinical_df_clean['SubjectID'].isin(site_subject_curr['SubjectID'])
            
        def to_int_sortable_id(s):
            return int(''.join([c for c in s if c.isdigit()]))
        def to_str_sortable_id(s):
            return ''.join([c for c in s if not c.isdigit()])
        
        clinical_df_clean = clinical_df_clean[subject_filter]
        
        clinical_df_clean['SortableIDint'] = clinical_df_clean['SubjectID'].apply(to_int_sortable_id)
        clinical_df_clean['SortableIDstr'] = clinical_df_clean['SubjectID'].apply(to_str_sortable_id)
        
        clinical_df_final = pd.concat([clinical_df_final, clinical_df_clean])
    
    clinical_df_final.sort_values(by=['Site','SortableIDstr', 'SortableIDint'], inplace=True)
    clinical_df_final.drop(columns=['SortableIDstr', 'SortableIDint'], inplace=True)
    
    clinical_df_final.to_csv(os.path.join(clinical_new_path, 'clinical_subject.csv'), index=False)
    

def load_clinical():
    '''
    Load the cleaned clinical data file
    
    return:
        DataFrame containing clinical data for all subjects from all sites
    '''
    csv_path = os.path.abspath(os.path.join(data_path, 'clinical_data', 'clinical_subject.csv'))
    final_df = pd.read_csv(csv_path)
    return final_df


if __name__ == "__main__":
    clean_clinical(overwrite=True)
