#%%
import sys

sys.path.append("/home/oliver/doc-forest_clis_ba_ws21/nice")
import os.path as op
from copy import copy, deepcopy

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from nice import read_markers

from config import chunk_seconds
from utils import reduce_to_epochs, reduce_to_scalars

sns.set_color_codes()

from config import get_reduction_params, reduction_functions, window, chunk_seconds

#%%

markers_folder = "../../data/processed/seeg_markers"
epoch_labels_path = markers_folder + "/" + "epoch_labels.csv"

w = window.replace(" ", "")
out_path = f"../../data/processed/seeg_features_epochs_{w}.csv"

df_epoch_labels = pd.read_csv(epoch_labels_path, index_col=[0, 1])

features = pd.DataFrame()
for i, patient in tqdm(list(enumerate(df_epoch_labels.index.get_level_values(0).unique()))):
    markers_file = markers_folder + "/" + f'/{patient}-markers.hdf5'
    if not op.exists(markers_file):
        raise ValueError('Please run compute_doc_forest_markers.py example first')
    markers = read_markers(markers_file)
    # summarize conscious epochs
    info_c = df_epoch_labels.loc[patient, "conscious"]
    epochs_df_c = reduce_to_epochs(markers, reduction_functions, info_c, window, chunk_seconds)
    # summarize unconscious epochs
    info_uc = df_epoch_labels.loc[patient, "unconscious"]
    epochs_df_uc = reduce_to_epochs(markers, reduction_functions, info_uc, window, chunk_seconds)
    if not i:
        epochs_df_uc.to_csv(out_path)
        epochs_df_c.to_csv(out_path, mode="a", header=False)
    else:
        epochs_df_uc.to_csv(out_path, mode="a", header=False)
        epochs_df_c.to_csv(out_path, mode="a", header=False)
    del markers, epochs_df_uc, epochs_df_c
