from datautils import load_corr_clinical
from fileutils import data_path
import os

reduced_ROI_path = os.path.join(data_path, 'atlas_clinical', 'RS_corr_148_merged_20210327.csv')

df = load_corr_clinical(clean_cols=False, remove_missing=False, reduce_ROI=True)

df.to_csv(reduced_ROI_path, index=False)
