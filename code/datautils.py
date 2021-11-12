import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression


pd.options.mode.chained_assignment = None  # default='warn'

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
    
    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        selected_rows = self.df.iloc[idx]
        
        data = selected_rows[self.data_columns]
        diagnosis = selected_rows[self.target_column]
        info = selected_rows[self.info_columns]
        clinical = selected_rows[self.clinical_columns]
        index = selected_rows.index

        sample = {'data':data, 'diagnosis':diagnosis, 'info':info, 
                  'clinical':clinical, 'index':index}
        
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
        
        return {'data':data, 'data_noisy':data_noisy, 'diagnosis':sample['diagnosis'],
                'info':sample['info'], 'clinical':sample['clinical'],
                'index':sample['index']}
        

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
        clinical = sample['clinical']
        if type(clinical) is pd.DataFrame:
            clinical_formatted = clinical.to_dict(orient='list')
        else:
            clinical_formatted = clinical.to_dict()
        sample_new = {'data':data_formatted, 
                      'diagnosis':diagnosis_formatted, 
                      'info':info_formatted, 'clinical':clinical_formatted}
        if 'data_noisy' in sample.keys():
            data_noisy = sample['data_noisy'].astype('float64')
            data_noisy_formatted = torch.from_numpy(data_noisy.to_numpy()).float()
            sample_new['data_noisy'] = data_noisy_formatted
        return sample_new
        

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


def remove_empty(df, columns=[], axis=1, how="all"):
    '''
    Removes all empty columns
    
    params:
        df: DataFrame
            The DataFrame to remove empty columns from
        columns: list
            The names of the subset of columns to remove if empty
        axis: int
            0 for rows
            1 for columns
        how: str
            "any": removes all rows(axis=0)/columns(axis=1) that have any empty cells
            "all": removes all rows(axis=0)/columns(axis=1) that have all empty cells
    '''
    df.replace("", np.nan, inplace=True)
    df[df.applymap(lambda x: type(x) is str and x.isspace())] = np.nan
    if columns:
        df.dropna(how=how, axis=axis, subset=columns, inplace=True)
    else:
        df.dropna(how=how, axis=axis, inplace=True)


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
    
    ROIs = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 137, 138, 139, 174, 175, 176, 177, 178, 179, 180, 181, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 251, 252, 256, 257, 258, 259, 260, 261, 262, 263, 264]
    def compare_ROI(c):
        if 'ROI' not in c:
            return True
        ROIx, ROIy = c.replace('ROI', '').split('-')
        return int(ROIx) in ROIs and int(ROIy) in ROIs
    ROI_filter = df.columns.to_series().apply(compare_ROI)
    df = df[df.columns[ROI_filter]]
    
    return df


def remove_redundant_ROI(df):
    '''
        Removes ROIs from the resting state correlation matrix 
        that are redundant e.g. ROI1-ROI1, ROI2-ROI2, ...
        
        params:
            df: DataFrame
                The DataFrame to apply the ROI removal
        
        return:
            DataFrame
                The DataFrame that the ROIs were removed for
    '''
    
    def compare_ROI(c):
        if 'ROI' not in c:
            return True
        ROIx, ROIy = c.replace('ROI', '').split('-')
        return int(ROIx) != int(ROIy)
    ROI_filter = df.columns.to_series().apply(compare_ROI)
    df = df[df.columns[ROI_filter]]
    
    return df


def preprocess_df(df, **kwargs):
    '''
        A utility function for scaling features and/or reducing the ROI if desired
        
        **kwargs:
            reduce_ROI: bool (default = False)
                Whether to apply ROI reduction
            remove_redundant_ROI: bool (default = False)
                Whether to remove redundant ROIs
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
    rem_red_ROI = kwargs.get('remove_redundant_ROI', False)
    drop_thresh = kwargs.get('drop_thresh', None)
    fill_empty = kwargs.get('fill_empty', None)
    regress_site = kwargs.get('regress_site', False)
    scale_feat = kwargs.get('scale_features', None)
    cols_to_ignore = kwargs.get('cols_to_ignore', [])
    
    if red_ROI:
        df = reduce_ROI(df)
        
    if rem_red_ROI:
        df = remove_redundant_ROI(df)
        
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
        elif fill_empty == "zero":
            df[cols] = df[cols].fillna(0)
    
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
                The percent from the training set to use for the validation set
        
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
            

def load_corr_clinical(csv_path, **kwargs):
    '''
    Loads the DataFrame containing RS correlation matrix and clinical data for
    subjects
    
    params:
        csv_path: str
            The path to the csv file containing the merged RS correlation matrix and clinical data
    
    **kwargs:
        patient_type: str (default = 'all')
            The patient population to load
        regress_site: bool (default = False)
            Whether to regress out the site
        clean_cols: bool (default = True)
            Whether to clean the Sex and Diagnosis columns or not
        remove_missing: bool (default = True)
            Whether to remove missing Age, Sex, and Diagnosis columns or not
        remove_empty_cols: bool (default = True)
            Whether to remove columns with all empty cells
        scale_features: str (default = None)
                Whether to scale features and if so, which scaler to use
        reduce_ROI: bool (default = False)
            Whether to apply ROI reduction
        remove_redundant_ROI: bool (default = True)
            Whether to remove redundant ROIs
            
    return: 
        DataFrame
            The correlation and clinical data for the patients of the
            desired patient types with preprocessing applied
    '''
    
    patient_type = kwargs.get('patient_type', 'all')
    patient_type = patient_type.lower()
    clean_cols_ = kwargs.get('clean_cols', True)
    remove_missing = kwargs.get('remove_missing', True)
    remove_empty_cols = kwargs.get('remove_empty_cols', True)
    kwargs['remove_redundant_ROI'] = kwargs.get('remove_redundant_ROI', True)
    
    diag_dict = {'Control':0, 'PTSD':1, 'Subthreshold':2}
    
    clinical_cols = ['Site', 'SubjectID', 'Age', 'Sex']

    chunksize = 10**3
    df = None

    for i,chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        
        if clean_cols_:
            chunk = clean_cols(chunk, sex=True, diag=True)
            diag_dict_local = diag_dict
        else:
            chunk = clean_cols(chunk, sex=False, diag=False)
            diag_dict_local = {k:k for k in diag_dict}
            
        if df is None:
            df = chunk
        else:
            df = pd.concat([df, chunk])
    
    if remove_missing:
        remove_empty(df, axis=0, how="any", columns=['Age', 'Sex', 'Diagnosis'])
    if remove_empty_cols:
        remove_empty(df, axis=1, how="all")
    
    patient_type = patient_type.lower()
    if patient_type != 'all':
        if patient_type == 'control':
            patient_filter = df['Diagnosis'] == diag_dict_local['Control']
            df = df[patient_filter]
        elif patient_type == 'ptsd':
            patient_filter = df['Diagnosis'] == diag_dict_local['PTSD']
            df = df[patient_filter]
        elif patient_type == 'subthreshold':
            patient_filter = df['Diagnosis'] == diag_dict_local['Subthreshold']
            df = df[patient_filter]
        elif patient_type == 'control_ptsd':
            patient_filter = df['Diagnosis'] != diag_dict_local['Subthreshold']
            df = df[patient_filter]
        else:
            raise ValueError("Invalid argument {} for kwarg 'patient_type'. Only accepts 'all', 'control', 'ptsd', 'control_ptsd', or 'subthreshold'".format(patient_type))
    
    df = preprocess_df(df, **kwargs)
    
    df['SubjectID'] = df['SubjectID'].apply(str)
    df['Age'] = df['Age'].astype(float)
    
    df.reset_index(drop=True, inplace=True)
    
    return df
    

def generate_datasets(csv_path, **kwargs):
    '''
    Generates the PatientDataSets of the desired format
    
    params:
        csv_path: str
            Path to csv file containing brain scan and clinical data
    
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
        remove_redundant_ROI: bool (default = True)
            Whether to remove redundant ROIs like ROI1-ROI1, ROI2-ROI2, etc given that the type is 'resting state'
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
        remove_missing: bool (default = True)
            Whether to remove missing Age, Sex, and Diagnosis columns or not
        remove_empty_cols: bool (default = True)
            Whether to remove columns with all empty cells
            
    
    return:
        PatientDataSet: if train_test_split is False
        PatientDataSet, PatientDataSet: if train_test_split is True 
    '''
    
    dataset_type = kwargs.get('dataset_type', 'RS')
    non_data_cols = kwargs.get('non_data_cols', [])
    
    kwargs['cols_to_ignore'] = non_data_cols
    
    if dataset_type == 'RS':
        df = load_corr_clinical(csv_path, **kwargs)
    else:
        raise ValueError("Invalid argument {} for kwarg 'type'. Only accepts 'RS'".format(dataset_type))
    
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