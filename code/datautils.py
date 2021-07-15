import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from atlas_to_file import load_patient_corr_264
from clinical_to_file import load_clinical
from clean_DTI_T1 import load_DTI_cleaned, load_T1_cleaned, load_T1_combat_cleaned
from fileutils import *

pd.options.mode.chained_assignment = None  # default='warn'

# Path for serialized combined correlation and clinical data
feather_path = os.path.abspath(os.path.join(serialized_path, 'atlas_clinical'))

    
class PatientDataSet(Dataset):
    '''
    PyTorch Dataset for holding the brain scan data and clinical data of patients
    and loading it for use in machine learning algorithms and PyTorch neural networks
    
    attributes:
        df: DataFrame 
            The dataframe containing correlation matrices and clinical data of patients
        dataset_type: str
            The type of brain scan data we are working with
        brain_columns: Index
            Column names containing the brain data
        clinical_columns: Index 
            Column names containing the clinical data
        data_columns: Index
            Column names containing the trainable data
        target_column: Index
            Column name of the target data
        info_columns: Index
            Column names of data containing additional, nontrainable data
        transform
            The transform(s) to be applied to the data
    '''
    
    def __init__(self, df, transform=None, dataset_type='RS', non_data_cols=[]):
        self.df = df
        self.dataset_type = dataset_type
        patient_cols = ['Site', 'SubjectID', 'Diagnosis', 'Age', 'Sex']
        brain_data_filter = ~(self.df.columns.to_series().isin(patient_cols + non_data_cols))
        self.brain_columns = self.df.columns[brain_data_filter]
        self.clinical_columns = self.df.columns[~brain_data_filter]
        info_cols = ['Site', 'SubjectID']
        target_col = ['Diagnosis']
        non_data = non_data_cols + info_cols + target_col
        data_col_filter = ~self.df.columns.isin(non_data)
        self.data_columns = self.df.columns[data_col_filter]
        target_col_filter = self.df.columns.isin(target_col)
        self.target_column = self.df.columns[target_col_filter]
        info_col_filter = self.df.columns.isin(info_cols)
        self.info_columns = self.df.columns[info_col_filter]
        self.transform = transform
        
        ##print(self.data_columns)
    
    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        selected_rows = self.df.iloc[idx]
        
        data = selected_rows[self.data_columns]
        diagnosis = selected_rows[self.target_column]
        info = selected_rows[self.info_columns]
        index = selected_rows.index

        sample = {'data':data, 'diagnosis':diagnosis, 'info':info, 'index':index}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    # Returns the trainable data of the entire dataset
    def get_X(self):
        data = self.df[self.data_columns]
        return data
    
    # Returns the target data of the entire dataset
    def get_Y(self):
        diagnosis = self.df[self.target_column]
        return diagnosis
        

class GaussianNoise:
    '''
    A transform to be applied to a PatientDataSet that applies gaussian 
    noise to it
    
    attributes:
        mean: float
            The mean of the gaussian noise
        std_dev: float
            The standard deviation of the gaussian noise
        brain_columns: Index
            The columns of the DataFrame, which hold the brain
            scan data to which the gaussian noise should be applied
    '''
    
    def __init__(self, mean, std_dev, brain_columns):
        self.mean = mean
        self.std_dev = std_dev
        self.brain_columns = brain_columns
        
    def __call__(self, sample):
        shape = sample['data'][self.brain_columns].shape
        noise = np.random.normal(self.mean, self.std_dev, shape)
        
        data = sample['data']
        data_noisy = data.copy()
        data_noisy[self.brain_columns] += noise
        
        return {'data':data, 'data_noisy':data_noisy, 'diagnosis':sample['diagnosis'], 'info':sample['info'], 'index':sample['index']}
        

class ToTorchFormat:
    '''
    A transform to be applied to a PatientDataSet that converts a
    given data sample to a PyTorch format
    '''
    
    def __call__(self, sample):
        data = sample['data'].astype('float64')
        data_formatted = torch.from_numpy(data.to_numpy()).float()
        diagnosis = sample['diagnosis'].astype('int64')
        diagnosis_formatted = torch.from_numpy(diagnosis.to_numpy())
        info = sample['info']
        if type(info) is pd.DataFrame:
            info_formatted = info.to_dict(orient='list')
        else:
            info_formatted = info.to_dict()
        sample_new = {'data':data_formatted, 
                      'diagnosis':diagnosis_formatted, 
                      'info':info_formatted}
        if 'data_noisy' in sample.keys():
            data_noisy = sample['data_noisy'].astype('float64')
            data_noisy_formatted = torch.from_numpy(data_noisy.to_numpy()).float()
            sample_new['data_noisy'] = data_noisy_formatted
        return sample_new


def match_corr_clinical(should_return=True):
    '''
    Matches the correlation matrices to the clinical data of patients
    and saves that as a csv
    
    params:
        should_return: bool (default = True)
            Whether to return the DataFrame containing the matched data
    
    return:
        (optional) The DataFrame containing the matched data
    '''
    
    corr_df = load_patient_corr_264()
    corr_df.dropna(axis=1, how='all', inplace=True)
    corr_df = corr_df.transpose()
    corr_df['Site'] = corr_df.index.to_series().apply(lambda s: s.split(" ")[0])
    corr_df['SubjectID'] = corr_df.index.to_series().apply(lambda s: s.split(" ")[1])
    corr_df.reset_index(drop=True, inplace=True)
    corr_df.columns.name = None
    clinical_df = load_clinical()
    final_df = corr_df.merge(clinical_df, on=['Site','SubjectID'])
    final_df[final_df.applymap(lambda x: type(x) is str and (x.isspace() or x == ""))] = np.nan
    cols_to_move = ['Site', 'SubjectID','Diagnosis', 'Age', 'Sex']
    col_order = cols_to_move + [col for col in final_df.columns if col not in cols_to_move]
    final_df = final_df[col_order]
    csv_path = os.path.join(data_path, 'atlas_clinical', 'corr_264_clinical.csv')
    final_df.to_csv(csv_path, index=False)
    
    ##print(len(final_df.index))
    
    merged_subj_df = final_df[['Site','SubjectID', 'Diagnosis']]
    csv_path = os.path.join(data_path, 'atlas_clinical', 'merged_subjects.csv')
    merged_subj_df.to_csv(csv_path, index=False)
    
    if should_return:
        return final_df
    else:
        return None
        

def merge_ica(should_return=True):
    merged_subj_path = os.path.join(data_path, 'atlas_clinical', 'merged_subjects.csv')
    merged_subj_df = pd.read_csv(merged_subj_path)
    merged_subj_df['SubjID'] = merged_subj_df['SubjectID'].apply(lambda s : s.replace("_", ""))
    ica_path = os.path.join(raw_data_path, 'ICA', 'RS_ICA_output.csv')
    ica_df = pd.read_csv(ica_path)
    ica_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    ica_df['SubjID'] = ica_df['SubjID'].apply(lambda s : s.replace("_", ""))
    final_df = pd.merge(merged_subj_df, ica_df, on=['Site', 'SubjID'], how='inner')
    final_df_test = pd.merge(merged_subj_df, ica_df, on=['Site', 'SubjID'], how='left', indicator='which')
    ##print(final_df_test[final_df_test['which'] == 'left_only'][['Site', 'SubjID']]) 
    final_df.drop(['SubjID'], axis=1, inplace=True)
    final_csv_path = os.path.join(data_path, 'ICA', 'ICA_merged.csv')
    final_df.to_csv(final_csv_path, index=False)
    
    if should_return:
        return final_df
    else:
        return None
    
        
'''
Dictionaries holding the maps for sex and diagnosis
'''
sex_dict = {'M':0, 'F':1} 
diag_dict = {'Control':0, 'PTSD':1, 'Subthreshold':2}

def clean_cols(df, subj=True, sex=True, diag=True):
    '''
    Cleans the SubjectID, Sex, and Diagnosis columns by mapping them
    to their desired values and types
    
    params:
        df: DataFrame
            The DataFrame to clean the columns for
        sex: bool (default = True)
            Whether to map the 'Sex' column to a number
        diag: bool (default = True)
            Whether to map the 'Diag' column to a number
            
    return:
        DataFrame
            The DataFrame with the columns cleaned
    '''
    
    if subj:
        df['SubjectID'] = df['SubjectID'].apply(str)
    if sex:
        df['Sex'] = df['Sex'].map(sex_dict)
    if diag:
        df['Diagnosis'] = df['Diagnosis'].map(diag_dict)
    
    return df
    

def scale_features(df, scaler_type='robust', cols_to_ignore = []):
    '''
        Scales all relevant feature columns using the specified scaler
        
        params:
            df: DataFrame
                The DataFrame to which to apply feature scaling
            scaler_type: str (default = 'robust')
                The string specifying the type of scaler to use
            cols_to_ignore: list (default = [])
                A list of column names to not apply scaling to
                
        return:
            DataFrame
                The DataFrame with scaled features
    '''
      
    if scaler_type.lower() == 'robust':
        scaler = RobustScaler()
    elif scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    elif scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise Exception("Invalid scaler type given. Only 'standard', 'robust', and 'minmax' are accepted.")
    cols_to_ignore = ['Site', 'SubjectID', 'Sex', 'Diagnosis'] + cols_to_ignore
    for col in df.columns[~df.columns.isin(cols_to_ignore)]:
        col_np = df[col].to_numpy().reshape(-1,1)
        scaler.fit(col_np)
        df[col] = scaler.transform(col_np)
        
    return df


def regress_out_site(df, cols_to_ignore=[]):
    '''
    Regresses out the site feature
    
    params:
        df: DataFrame
            The DataFrame to regress out the site for
        cols_to_ignore: list (default = [])
            A list of columns to not regress out the site for
            
    return:
        DataFrame
            The DataFrame for which the site was regressed out
    '''
        
    unique_sites = df['Site'].unique()
    df_regress_cols = df.columns[~df.columns.isin(['Site', 'SubjectID', 'Diagnosis'] + cols_to_ignore)]
    sites = df['Site'].unique()
    site_df = pd.DataFrame(columns=sites)
    for site in sites:
        site_df[site] = df['Site'] == site
    site_df = site_df.astype(int)
    lr = LinearRegression()
    
    last_percent = 0
    print(last_percent, '%')
    
    for i,col in enumerate(df_regress_cols):
        
        curr_percent = int(i/len(df_regress_cols)*100)
        if curr_percent != last_percent:
            print(curr_percent, '%')
        last_percent=curr_percent
        
        Y = df[col]
        X = site_df
        lr.fit(X, Y)
        Y_pred = lr.predict(X)
        resid = Y - Y_pred
        df[col] = resid
    
    return df
        

def reduce_ROI(df):
    '''
        Removes ROIs from the resting state correlation matrix 
        that aren't hypothesized in relating to PTSD
        
        params:
            df: DataFrame
                The DataFrame to apply the ROI reduction to
        
        return:
            DataFrame
                The DataFrame that the ROIs were reduced for
    '''
    
    reduced_ROI_fpath = os.path.join(data_path, 'atlas', 'ROI_reduced.txt')
    with open(reduced_ROI_fpath) as f:
        ROIs = [s.replace('\n', '') for s in f.readlines()]
    def compare_ROI(c):
        if 'ROI' not in c:
            return True
        ROIx, ROIy = c.replace('ROI', '').split('-')
        return ROIx in ROIs and ROIy in ROIs
    ROI_filter = df.columns.to_series().apply(compare_ROI)
    df = df[df.columns[ROI_filter]]
    
    return df
    
    
def remove_empty(df, columns):
    df.replace("", np.nan, inplace=True)
    df[df.applymap(lambda x: type(x) is str and x.isspace())] = np.nan
    df.dropna(subset=columns, inplace=True)
    ##print(df[columns].columns[df[columns].isna().any(axis=0)].tolist())
    

def preprocess_df(df, **kwargs):
    '''
        A utility function for scaling features and/or reducing the ROI if desired
        
        **kwargs:
            reduce_ROI: bool (default = False)
                Whether to apply ROI reduction
            drop_thresh: float (default = None)
                If not None, drop rows that don't exceed a proportion of
                drop_thresh percent non-empty cells
            fill_empty: bool (default = None)
                Whether to use the average column value to fill in empty
                cells
            regress_site: bool (default = False)
                Whether to regress out the site information
            scale_features: str (default = None)
                Whether to scale features and if so, which scaler to use
        
        return:
            DataFrame
                The DataFrame with desired preprocessing applied
    '''
    
    red_ROI = kwargs.get('reduce_ROI', False)
    drop_thresh = kwargs.get('drop_thresh', None)
    fill_empty = kwargs.get('fill_empty', None)
    regress_site = kwargs.get('regress_site', False)
    scale_feat = kwargs.get('scale_features', None)
    cols_to_ignore = kwargs.get('cols_to_ignore', [])
    
    if red_ROI:
        df = reduce_ROI(df)
        
    if drop_thresh:
        df = df.replace('', np.nan)
        df.dropna(thresh=int(drop_thresh*len(df.columns)), inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    if fill_empty:
        df = df.replace('', np.nan)
        cols_to_ignore_tmp = cols_to_ignore + ['Site', 'SubjectID','Diagnosis', 'Age', 'Sex']
        cols = df.columns[~df.columns.isin(cols_to_ignore_tmp)]
        if fill_empty == "mean":
            means = pd.Series(df[cols].mean(axis=0, skipna=True), index=cols)
            df[cols] = df[cols].fillna(means)
    
    if regress_site:
        df = regress_out_site(df, cols_to_ignore=cols_to_ignore)
        
    if scale_feat:
        df = scale_features(df, scale_feat, cols_to_ignore=cols_to_ignore)
        
    return df
    

def split_df(df, **kwargs):
    '''
        A function for splitting the DataFrame either by choosing a selected number of rows
        or by doing a train test split by percent
        
        **kwargs:
            n_rows: int (default = None)
                If not None, the number of rows to take from the DataFrame
            randomize: bool (default = False)
                Whether to shuffle the rows
            random_seed: int (deafult =  None)
                If not None, the random seed to plug into the random shuffle
            train_test_split: bool (default = False)
                Whether to perform a train test split by percent
            percent_split: float(default = None)
                The percent to split by between 0.0 and 1.0
            validation_split: float(default = None)
                The amount from the training set to use for the validation set
        
        return:
            DataFrame: if train_test_split is False
            DataFrame, DataFrame: if train_test_split is True
    '''
    
    n_rows = kwargs.get('n_rows', None)
    randomize = kwargs.get('randomize', False)
    random_seed = kwargs.get('random_seed', None)
    train_test_split = kwargs.get('train_test_split', False)
    percent_split = kwargs.get('percent_split', None)
    validation_split = kwargs.get('validation_split', None)
    
    if n_rows:
        n_rows = min(n_rows, len(df.index)-1)
    
    if not randomize:
        if train_test_split:
            if percent_split:
                if n_rows:
                    split = int(percent_split*n_rows)
                    train_df = df.iloc[0:split]
                    test_df = df.iloc[split:n_rows]
                else:
                    split = int(percent_split*(len(df.index)-1))
                    train_df = df.iloc[0:split]
                    test_df = df.iloc[split:]
                if validation_split:
                    split = int(validation_split*(len(train_df.index)-1))
                    val_df = train_df.iloc[0:split]
                    train_df = train_df.iloc[split:]
            else:
                raise Exception("Wanted train test split, but no split was given. Must set 'percent_split' as an argument.")
            
            if not validation_split:
                return train_df, test_df
            else:
                return train_df, val_df, test_df
            
        else:
            if percent_split:
                if n_rows:
                    split = int(percent_split*n_rows)
                else:
                    split = int(percent_split*(len(df.index)-1))
                df = df.iloc[0:split]
            elif n_rows:
                df = df.iloc[0:nrows]
            
            return df
            
    else:
        if random_seed:
            np.random.seed(random_seed)
        
        if train_test_split:
            if percent_split:
                if n_rows:
                    split = int(percent_split*n_rows)
                    df_indices = np.random.choice(df.index, n_rows)
                else:
                    split = int(percent_split*(len(df.index)-1))
                    df_indices = np.random.choice(df.index, len(df.index))
                train_df = df.loc[df_indices[0:split]]
                test_df = df.loc[df_indices[split:]]
                
                if validation_split:
                    split = int(validation_split*(len(train_df.index)-1))
                    val_df = train_df.iloc[0:split]
                    train_df = train_df.iloc[split:]
            else:
                raise Exception("Wanted train test split, but no split was given. Must set 'percent_split' as an argument.")
            
            if not validation_split:
                return train_df, test_df
            else:
                return train_df, val_df, test_df
            
        else:
            if percent_split:
                if n_rows:
                    split = int(percent_split*n_rows)
                else:
                    split = int(percent_split*(len(df.index)-1))
                df_indices = np.random.choice(df.index, split)
                df = df.loc[df_indices]
            elif n_rows:
                df_indices = np.random.choice(df.index, n_rows)
                df = df.loc[df_indices]
            else:
                df_indices = np.random.choice(df.index, len(df.index))
                df = df.loc[df_indices]
            
            return df


def load_corr_clinical(load_corr=True, load_clinical=True, **kwargs):
    global diag_dict
    '''
    Loads the DataFrame containing correlation matrix and clinical data for
    subjects
    
    params:
        load_corr: bool (default = True)
            Whether to load the correlation matrix columns
        load_clinical: bool (default = True)
            Whether to load the clinical data columns
    
    **kwargs:
        patient_type: str (default = 'all')
            The patient population to load
        regress_site: bool (default = False)
            Whether to regress out the site
        clean_cols: bool (default = True)
            Whether to clean the Sex and Diagnosis columns or not
        remove_missing: bool (default = True)
            Whether to remove missing Age, Sex, and Diagnosis columns or not
        scale_features: str (default = None)
                Whether to scale features and if so, which scaler to use
        reduce_ROI: bool (default = False)
            Whether to apply ROI reduction
            
    return: 
        DataFrame
            The correlation and/or clinical data for the patients of the
            desired patient types with preprocessing applied
    '''
    
    patient_type = kwargs.get('patient_type', 'all')
    patient_type = patient_type.lower()
    clean_cols_ = kwargs.get('clean_cols', True)
    remove_missing = kwargs.get('remove_missing', True)
    
    if not load_corr and not load_clinical:
        return None
    
    clinical_cols = ['Site', 'SubjectID', 'Age', 'Sex']
    
    csv_path = os.path.join(data_path, 'atlas_clinical', 'corr_264_clinical.csv')

    chunksize = 10**3
    df = None

    for i,chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        
        if not load_clinical:
            chunk = chunk[chunk.columns[~chunk.columns.isin(clinical_cols)]]
        elif not load_corr:
            chunk = chunk[chunk.columns[chunk.columns.isin(clinical_cols)]]
        
        if clean_cols_:
            chunk = clean_cols(chunk, sex=True, diag=True)
            diag_dict_local = diag_dict
        else:
            chunk = clean_cols(chunk, sex=False, diag=False)
            diag_dict_local = {k:k for k in diag_dict}
        
        if patient_type != 'all':
            if patient_type == 'control':
                patient_filter = chunk['Diagnosis'] == diag_dict_local['Control']
                chunk = chunk[patient_filter]
            elif patient_type == 'ptsd':
                patient_filter = chunk['Diagnosis'] == diag_dict_local['PTSD']
                chunk = chunk[patient_filter]
            elif patient_type == 'subthreshold':
                patient_filter = chunk['Diagnosis'] == diag_dict_local['Subthreshold']
                chunk = chunk[patient_filter]
            elif patient_type == 'control_ptsd':
                patient_filter = chunk['Diagnosis'] != diag_dict_local['Subthreshold']
                chunk = chunk[patient_filter]
            else:
                raise ValueError("Invalid argument {} for kwarg 'patient_type'. Only accepts 'all', 'control', 'ptsd', 'control_ptsd', or 'subthreshold'".format(patient_type))
            
        if df is None:
            df = chunk
        else:
            df = pd.concat([df, chunk])
    
    if remove_missing:
        remove_empty(df, columns=['Age', 'Sex', 'Diagnosis'])
    
    df = preprocess_df(df, **kwargs)
    
    df['SubjectID'] = df['SubjectID'].apply(str)
    df['Age'] = df['Age'].astype(float)
    
    df.reset_index(drop=True, inplace=True)
    
    return df
            
            
def serialize_corr_clinical(overwrite=False, **kwargs):
    '''
    Serializes the correlation matrix and clinical data combined data for fast loading
    
    params:
        overwrite: bool (default = False)
            Whether to overwrite the serialized data if it exists
    
    **kwargs:
        regress_site: bool (default = False)
            Whether to regress the site out or not
    '''
    
    regress_site = kwargs.get('regress_site', False)
    
    orig_feather_name = os.path.join(feather_path, 'corr_264_clinical.ftr')
    site_regressed_feather_name = os.path.join(feather_path, 'corr_264_site_regressed.ftr')
    
    if not regress_site:
        if overwrite or not os.path.exists(orig_feather_name):
            df = load_corr_clinical(**kwargs)
            df.to_feather(orig_feather_name)
    else:
        if overwrite or not os.path.exists(site_regressed_feather_name):
            if not os.path.exists(orig_feather_name):
                raise FileNotFoundError("The original serialized file does not yet exist. Please call 'serialize_corr_clinical' with 'preprocess' set to 'False' first.")
            df = pd.read_feather(orig_feather_name)
            df = regress_out_site(df)
            df.to_feather(site_regressed_feather_name)
    
    
def prepare_serialized(overwrite_orig=False, overwrite_site_regressed=False, **kwargs):
    '''
    Prepares and saves the different versions of serialized data we want
    
    params:
        overwrite_orig : bool (default = False)
            Whether to overwrite the not site-regressed serialized data
        overwrite_site_regressed : bool (default = False)
            Whether to overwrite the site-regressed serialized data
    '''
    
    if not os.path.exists(feather_path):
        os.makedirs(feather_path)
    serialize_corr_clinical(overwrite=overwrite_orig, **kwargs)
    # Want to serialize regressed site because the operation is slow
    # so it is beneficial to have that version stored on its own
    serialize_corr_clinical(overwrite=(overwrite_orig or overwrite_site_regressed), regress_site=True, **kwargs)
    
    
def load_corr_clinical_serialized(load_corr=True, load_clinical=True, **kwargs):
    '''
    Loads the serialized data containing correlation matrix and clinical data for
    subjects as a DataFrame
    
    params:
        load_corr: bool (default = True)
            Whether to load the correlation matrix columns
        load_clinical: bool (default = True)
            Whether to load the clinical data columns
    
    **kwargs:
        patient_type: str (default = 'all')
            The patient population to load
        regress_site: bool (default = False)
            Whether to regress out the site
        scale_features: str (default = None)
                Whether to scale features and if so, which scaler to use
        reduce_ROI: bool (default = False)
            Whether to apply ROI reduction
            
    return: 
        DataFrame
            The correlation and/or clinical data for the patients of the
            desired patient types with preprocessing applied
    '''
    
    regress_site = kwargs.get('regress_site', False)   
    patient_type = kwargs.get('patient_type', 'all')
    patient_type = patient_type.lower()
    
    if not load_corr and not load_clinical:
        return None
    
    clinical_cols = ['Site', 'SubjectID', 'Age', 'Sex']
    
    orig_feather_name = os.path.join(feather_path, 'corr_264_clinical.ftr')
    site_regressed_feather_name = os.path.join(feather_path, 'corr_264_site_regressed.ftr')
    
    if not regress_site:
        feather_name = orig_feather_name
    else:
        feather_name = site_regressed_feather_name
    
    # Do not want to prepare the serialized data inside this function if it does not exist, so instead
    # throw an error prompting first the creation of the serialized data with prepare_serialized
    if not os.path.exists(feather_name):
        raise FileNotFoundError("The serialized file '{}' does not exist. Please call perpare_serialized first to serialize data.".format(feather_name))
    
    df = pd.read_feather(feather_name)
            
    if not load_clinical:
        df = df[df.columns[~df.columns.isin(clinical_cols)]]
    elif not load_corr:
        df = df[df.columns[df.columns.isin(clinical_cols)]]
    
    if patient_type != 'all':
        if patient_type == 'control':
            patient_filter = df['Diagnosis'] == diag_dict['Control']
            df = df[patient_filter]
        elif patient_type == 'ptsd':
            patient_filter = df['Diagnosis'] == diag_dict['PTSD']
            df = df[patient_filter]
        elif patient_type == 'subthreshold':
            patient_filter = df['Diagnosis'] == diag_dict['Subthreshold']
            df = df[patient_filter]
        elif patient_type == 'control_ptsd':
            patient_filter = df['Diagnosis'] != diag_dict['Subthreshold']
            df = df[patient_filter]
        else:
            raise ValueError("Invalid argument {} for kwarg 'patient_type'. Only accepts 'all', 'control', 'ptsd', 'control_ptsd', or 'subthreshold'".format(patient_type))
    
    df = preprocess_df(df, **kwargs)
    
    return df
    

def load_DTI(**kwargs):
    
    patient_type = kwargs.get('patient_type', 'all')
    patient_type = patient_type.lower()
    remove_missing = kwargs.get('remove_missing', True)
    
    df = load_DTI_cleaned()
    DTI_diag_map = {'TEHC':'Control', 'HC':'Control', 'SubThresh':'Subthreshold', 'PTSD':'PTSD', 'Control':'Control'}
    df['Diagnosis'] = df['Diagnosis'].map(DTI_diag_map)
    
    df = clean_cols(df)
    
    if patient_type != 'all':
        if patient_type == 'control':
            patient_filter = df['Diagnosis'] == diag_dict['Control']
            df = df[patient_filter]
        elif patient_type == 'ptsd':
            patient_filter = df['Diagnosis'] == diag_dict['PTSD']
            df = df[patient_filter]
        elif patient_type == 'subthreshold':
            patient_filter = df['Diagnosis'] == diag_dict['Subthreshold']
            df = df[patient_filter]
        elif patient_type == 'control_ptsd':
            patient_filter = df['Diagnosis'] != diag_dict['Subthreshold']
            df = df[patient_filter]
        else:
            raise ValueError("Invalid argument {} for kwarg 'patient_type'. Only accepts 'all', 'control', 'ptsd', 'control_ptsd', or 'subthreshold'".format(patient_type))
    
    if remove_missing:
        remove_empty(df, columns=['Age', 'Sex', 'Diagnosis'])
    
    df = preprocess_df(df, **kwargs)
    
    df['SubjectID'] = df['SubjectID'].apply(str)
    df['Age'] = df['Age'].astype(float)
    
    return df
    
    
def load_T1(**kwargs):
    
    patient_type = kwargs.get('patient_type', 'all')
    patient_type = patient_type.lower()
    remove_missing = kwargs.get('remove_missing', True)
    
    df = load_T1_cleaned()
    T1_diag_map = {'TEHC':'Control', 'HC':'Control', 'SubThresh':'Subthreshold', 'PTSD':'PTSD', 'Control':'Control'}
    df['Diagnosis'] = df['Diagnosis'].map(T1_diag_map)
    
    df = clean_cols(df)
    
    if patient_type != 'all':
        if patient_type == 'control':
            patient_filter = df['Diagnosis'] == diag_dict['Control']
            df = df[patient_filter]
        elif patient_type == 'ptsd':
            patient_filter = df['Diagnosis'] == diag_dict['PTSD']
            df = df[patient_filter]
        elif patient_type == 'subthreshold':
            patient_filter = df['Diagnosis'] == diag_dict['Subthreshold']
            df = df[patient_filter]
        elif patient_type == 'control_ptsd':
            patient_filter = df['Diagnosis'] != diag_dict['Subthreshold']
            df = df[patient_filter]
        else:
            raise ValueError("Invalid argument {} for kwarg 'patient_type'. Only accepts 'all', 'control', 'ptsd', 'control_ptsd', or 'subthreshold'".format(patient_type))
    
    if remove_missing:
        remove_empty(df, columns=['Age', 'Sex', 'Diagnosis'])
    
    df = preprocess_df(df, **kwargs)
    
    df['SubjectID'] = df['SubjectID'].apply(str)
    df['Age'] = df['Age'].astype(float)
    
    return df


def load_T1_combat(**kwargs):
    
    patient_type = kwargs.get('patient_type', 'all')
    patient_type = patient_type.lower()
    remove_missing = kwargs.get('remove_missing', True)
    
    df = load_T1_combat_cleaned()
    T1_diag_map = {'TEHC':'Control', 'HC':'Control', 'SubThresh':'Subthreshold', 'PTSD':'PTSD', 'Control':'Control'}
    df['Diagnosis'] = df['Diagnosis'].map(T1_diag_map)
    
    df = clean_cols(df, subj=False, sex=False)
    
    if patient_type != 'all':
        if patient_type == 'control':
            patient_filter = df['Diagnosis'] == diag_dict['Control']
            df = df[patient_filter]
        elif patient_type == 'ptsd':
            patient_filter = df['Diagnosis'] == diag_dict['PTSD']
            df = df[patient_filter]
        elif patient_type == 'subthreshold':
            patient_filter = df['Diagnosis'] == diag_dict['Subthreshold']
            df = df[patient_filter]
        elif patient_type == 'control_ptsd':
            patient_filter = df['Diagnosis'] != diag_dict['Subthreshold']
            df = df[patient_filter]
        else:
            raise ValueError("Invalid argument {} for kwarg 'patient_type'. Only accepts 'all', 'control', 'ptsd', 'control_ptsd', or 'subthreshold'".format(patient_type))
    
    if remove_missing:
        remove_empty(df, columns=['Site', 'Diagnosis'])
    
    site_diag_filt = df.columns.isin(["Site", "Diagnosis"])
    df[df.columns[~site_diag_filt]] = df[df.columns[~site_diag_filt]].astype(np.float64)
    
    for col in df.columns:
        if not (df[col].dtype == np.dtype("float64") or df[col].dtype == np.dtype("int64")):
            print(col)
    
    df = preprocess_df(df, **kwargs)
    
    return df


def generate_datasets(load_corr=True, load_clinical=True, load_serialized=True, **kwargs):
    '''
    Generates the PatientDataSets of the desired format
    
    params:
        load_corr : bool (default = True)
            Whether to load the correlation matrix part of the data
        load_clinical : bool (default = True)
            Whether to load the clinical part of the data
        load_serialized : bool (default = True)
            Whether to load the serialized data version of the data for speed efficiency if it exists
    
    **kwargs:
        dataset_type: str (default = 'RS')
            Keeps track of what sort of dataset we are working with
        non_data_cols: list (default = [])
            Additional columns that should not be considered data
        patient_type: str (default = 'all')
            The patient population to load
        regress_site: bool (default = False)
            Whether to regress out the site
        scale_features: str (default = None)
                Whether to scale features and if so, which scaler to use
        reduce_ROI: bool (default = False)
            Whether to apply ROI reduction given that type is 'resting_state'
        n_rows: int (default = None)
            If not None, the number of rows to take from the DataFrame
        randomize: bool (default = False)
            Whether to shuffle the rows
        random_seed: int (deafult =  None)
            If not None, the random seed to plug into the random shuffle
        train_test_split: bool (default = False)
            Whether to perform a train test split by percent
        percent_split: float (default = None)
            The percent to split by between 0.0 and 1.0
    
    return:
        PatientDataSet: if train_test_split is False
        PatientDataSet, PatientDataSet: if train_test_split is True 
    '''
    
    dataset_type = kwargs.get('dataset_type', 'RS')
    non_data_cols = kwargs.get('non_data_cols', [])
    
    kwargs['cols_to_ignore'] = non_data_cols
    
    if dataset_type == 'RS':
        if load_serialized:
            df = load_corr_clinical_serialized(load_corr=load_corr, load_clinical=load_clinical, **kwargs)
        else:
            df = load_corr_clinical(load_corr, load_clinical, **kwargs)
    elif dataset_type == 'DTI':
        df = load_DTI(**kwargs)
    elif dataset_type == 'T1':
        df = load_T1(**kwargs)
    elif dataset_type == 'T1_combat':
        df = load_T1_combat(**kwargs)
    else:
        raise ValueError("Invalid argument {} for kwarg 'type'. Only accepts 'RS', 'DTI', or 'T1'".format(dataset_type))
    
    dfs = split_df(df, **kwargs)
    
    if type(dfs) is tuple:
        if len(dfs) == 2:
            train_dataset = PatientDataSet(dfs[0], dataset_type=dataset_type, non_data_cols=non_data_cols)
            test_dataset = PatientDataSet(dfs[1], dataset_type=dataset_type, non_data_cols=non_data_cols)
            return train_dataset, test_dataset
        elif len(dfs) == 3:
            train_dataset = PatientDataSet(dfs[0], dataset_type=dataset_type, non_data_cols=non_data_cols)
            val_dataset = PatientDataSet(dfs[1], dataset_type=dataset_type, non_data_cols=non_data_cols)
            test_dataset = PatientDataSet(dfs[2], dataset_type=dataset_type, non_data_cols=non_data_cols)
            return train_dataset, val_dataset, test_dataset
    else:
        return PatientDataSet(dfs, dataset_type=dataset_type, non_data_cols=non_data_cols)


if __name__ == "__main__":
    match_corr_clinical(False)
    '''
    prepare_serialized(overwrite_orig=True, overwrite_site_regressed=True, drop_thresh=0.8, fill_empty="mean")
    '''
    ##merge_ica(False)
    ##print(load_T1_combat())
