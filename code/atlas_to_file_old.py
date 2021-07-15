import os
import numpy as np
import pandas as pd

from fileutils import *


# The filepath for the connectivity data
atlas_con_path = os.path.join(raw_data_path, "atlas_connectivity", "unzipped")


def site_subject(should_return=True):
    '''
    Generate a file containing subject IDs and their respective sites
    
    params:
        should_return: bool (default = True)
            Whether to return the completed pandas dataframe containing all the subjects and their sites
    
    return:
        (optional) DataFrame containing subjects and their sites
    '''
    colnames = ['Site', 'SubjectID']
    df_dict = {c : [] for c in colnames}
    
    # Iterate over all sites in the connectivity folder and add all of their
    for site in os.listdir(atlas_con_path):
        # Special case for Toledo, where there are multiple subfolders before getting to_csv
        # connectivity data
        if site == "Toledo":
            subfolders = os.listdir(os.path.join(atlas_con_path, site))
            for subfolder in subfolders:
                site_sub = site + '_' + subfolder
                subjects = os.listdir(os.path.join(atlas_con_path, site, subfolder))
                sites = [site] * len(subjects)
                df_dict['Site'] += sites
                df_dict['SubjectID'] += subjects
        else:
            subjects = os.listdir(os.path.join(atlas_con_path, site))
            # Special case for Capetown, where _ needs to be removed
            if site == "Capetown":
                subjects = [str(s).replace('_','') for s in subjects]
            # Special case for Columbia, where leading 0s need to be removed
            elif site == "Columbia":
                subjects = [str(s).lstrip('0') for s in subjects]
            sites = [site] * len(subjects)
            df_dict['Site'] += sites
            df_dict['SubjectID'] += subjects
    
    # Create the dataframe using the dictionary of subjects and sites
    df = pd.DataFrame.from_dict(df_dict)
    
    # Sorting alphanumerically by Subject ID
    def to_int_sortable_id(s):
        return int(''.join([c for c in s if c.isdigit()]))
    def to_str_sortable_id(s):
        return ''.join([c for c in s if not c.isdigit()])
    
    df['SortableIDint'] = df['SubjectID'].apply(to_int_sortable_id)
    df['SortableIDstr'] = df['SubjectID'].apply(to_str_sortable_id)
    
    df.sort_values(by=['Site', 'SortableIDstr', 'SortableIDint'], inplace=True)

    df.drop(columns=['SortableIDstr', 'SortableIDint'], inplace=True)
    
    csv_path = os.path.join(data_path, 'site_subject.csv')
    df.to_csv(csv_path, index=False)
    
    if should_return:
        return df
    else:
        return None
        


def load_site_subject():
    '''
    Load the site-subject csv as a pandas DataFrame
    
    return:
        DataFrame containing subjects and their sites
    '''
    csv_path = os.path.join(data_path, 'site_subject.csv')
    return pd.read_csv(csv_path)



def patient_corr_264(should_return=True):
    '''
    Maps the patients to their correlation matrices
    
    params:
        should_return: bool (default = True)
            Whether to return the completed pandas dataframe containing all the subjects and their correlation matrices
    
    return:
        (optional) DataFrame containing subjects and their correlation matrices
    '''
    fname = "corr_matrix_Power264.csv"
    df_dict = {}
    
    # Generating the index lables for the upper triangular ROI correlation matrix
    ROI_indices = [] 
    for i in range(1,264+1):
        ROI_indices.append([])
        for j in range(1, 264+1):
            ROI_indices[i-1].append("ROI{}-ROI{}".format(i, j))
    ROI_indices = np.array(ROI_indices)
    ROI_indices = ROI_indices[np.triu_indices(264)]
    mat_empty = [None]*len(ROI_indices)
    df_dict['ROI'] = ROI_indices
    
    # Go through all the sites
    for site in os.listdir(atlas_con_path):
        print(site)
        sitepath = os.path.join(atlas_con_path, site)
        # Special case for Toledo since it has subdirectories for additional sites
        if site == "Toledo":
            subfolders = os.listdir(sitepath)
            for subfolder in subfolders:
                subfolderpath = os.path.join(sitepath, subfolder)
                subjects = os.listdir(subfolderpath)
                for subject in subjects:
                    colname = site + " " + subject
                    subjectpath = os.path.join(subfolderpath, subject)
                    if fname in os.listdir(subjectpath):
                        subject_cor = pd.read_csv(os.path.join(subjectpath, fname), header=None)
                        subject_cor = subject_cor.to_numpy()
                        # Adjusting for correlation matrices missing rows at the end
                        if(len(subject_cor) < 264):
                            subject_cor_new = np.zeros((264, 264))
                            subject_cor_new[:subject_cor.shape[0], :subject_cor.shape[1]] = subject_cor
                            subject_cor = subject_cor_new
                        subject_triu = list(subject_cor[np.triu_indices(len(subject_cor))])
                        df_dict[colname] = subject_triu
                    else:
                        df_dict[colname] = mat_empty
                        
                        
        else:
            sitepath = os.path.join(atlas_con_path, site)
            subjects = os.listdir(sitepath)
                
            for subject in subjects:
                # Special case for Capetown, where _ needs to be removed
                if site == "Capetown":
                    subject_id = str(subject).replace('_','')
                # Special case for Columbia, where leading 0s need to be removed
                elif site == "Columbia":
                    subject_id = str(subject).lstrip('0')
                else:
                    subject_id = str(subject)
                colname = site + " " + subject_id
                subjectpath = os.path.join(sitepath, subject)
                # If there is matching subject and correlation matrix
                # Map them to one another in the DataFrame
                if fname in os.listdir(subjectpath):
                    '''
                    if site == "Duke":
                        print("Subject", subject, "is ok.")
                    '''
                    subject_cor = pd.read_csv(os.path.join(subjectpath, fname), header=None)
                    subject_cor = subject_cor.to_numpy()
                    # Adjusting for correlation matrices missing rows at the end
                    if(len(subject_cor) < 264):
                        subject_cor_new = np.zeros((264, 264))
                        subject_cor_new[:subject_cor.shape[0], :subject_cor.shape[1]] = subject_cor
                        subject_cor = subject_cor_new
                    subject_triu = list(subject_cor[np.triu_indices(len(subject_cor))])
                    df_dict[colname] = subject_triu
                # Otherwise, keep the correlation matrix empty
                else:
                    '''
                    if site == "Duke":
                        print("Subject", subject, "is not ok.")
                    '''
                    df_dict[colname] = mat_empty
                    
    
    df = pd.DataFrame.from_dict(df_dict)
    df.set_index('ROI', inplace=True)
    csv_path = os.path.join(data_path, 'atlas', 'patient_corr_264.csv')
    df.to_csv(csv_path, index=True)
    
    if should_return:
        return df
    else:
        return None
        

def load_patient_corr_264():
    '''
    Load the correlation matrix csv as a pandas DataFrame
    
    return:
        DataFrame containing subjects and their correlation matrices
    '''
    csv_path = os.path.abspath(os.path.join(data_path, 'atlas', 'patient_corr_264.csv'))
    final_df = pd.read_csv(csv_path, index_col='ROI')
    return final_df


if __name__ == "__main__":
    site_subject(should_return=False)
    patient_corr_264(should_return=False)
    
