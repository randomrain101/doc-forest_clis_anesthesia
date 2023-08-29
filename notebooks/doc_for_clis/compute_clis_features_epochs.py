#%%
import sys
sys.path.append("/home/oliver/doc-forest_clis_ba_ws21/nice")
from copy import deepcopy, copy

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
import os.path as op

import mne

from config import base_psd, get_reduction_params, marker_list, reduction_functions, window, chunk_seconds
from utils import reduce_to_epochs, reduce_to_scalars, reduce_to_epochs_clis

from nice import Markers, read_markers

import seaborn as sns
sns.set_color_codes()

#%%
#
# --- compute clis markers ---
#
import mne
from mne.io import read_raw_fieldtrip

w = window.replace(" ", "")

# paths must be preordered by time
paths = [
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316003431-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316013431-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316023431-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316033431-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316043432-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316053432-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316063432-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316073432-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316083432-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316093432-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316103432-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316113432-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316123433-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316133433-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316143433-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316153433-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316163433-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316173433-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316183433-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316193433-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316203434-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316213434-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316223434-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat",
    "/home/oliver/doc-forest_clis_ba_ws21/data/raw/ecog_data/20080316233434-ECOGGR-BrainAmpMRx2-ref=G102-gnd=S032.mat"
]
# sort by timestamp contained in filename
paths.sort(key=lambda x: x.split("/")[-1].split("-")[0])
info = None

mc_parent = Markers(marker_list)

epochs_list = []

#%%
#raw = read_raw_fieldtrip(paths[0], info=info, data_name="eegdata")
#for i in range(0, 2):
for i in range(0, 24):
    #temp = read_raw_fieldtrip(paths[i], info=info, data_name="eegdata");
    #raw.append(temp)
    raw = read_raw_fieldtrip(paths[i], info=info, data_name="eegdata");

    #%%
    # get epochs
    annotations = mne.Annotations(
        onset=[raw.times[0]],
        duration=[raw.times[-1]],
        description=["unknown state"]
    )
    raw = raw.set_annotations(annotations)
    # nice only accepts eeg or meg channel types
    #raw = raw.set_channel_types({channel: "ecog" for channel in raw.info["ch_names"]})
    raw = raw.set_channel_types({channel: "eeg" for channel in raw.info["ch_names"]})

    #%%
    from config import chunk_seconds
    event_id = {"unknown state": 2}
    events = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=chunk_seconds)

    # create epochs
    epochs = mne.Epochs(raw, events[0], event_id, - chunk_seconds, 0, preload=True)

    #%%
    base_psd.fit(epochs)
    mc = Markers(marker_list)
    mc.fit(epochs);

    timestamp_start = pd.Timestamp(paths[i].split("/")[-1].split("-")[0])
    timestamp_end = timestamp_start + pd.Timedelta("1 hour")
    out_index = pd.DatetimeIndex(pd.date_range(timestamp_start, timestamp_end, periods=len(epochs)))
    df_epochs = reduce_to_epochs_clis(mc, reduction_functions, out_index)
    epochs_list.append(df_epochs)
    print(df_epochs.head())


raw_path = f"../../data/processed/clis_epochs_{w}.csv"
pd.concat(epochs_list, axis=0).to_csv(raw_path)

#%%
#
# --- reduce to epochs ---
#
out_path = f"../../data/processed/clis_features_epoch_{w}.csv"

#%%
df_epochs_features = pd.read_csv(raw_path, index_col=0, header=[0, 1, 2])

timestamp_start = pd.Timestamp(paths[0].split("/")[-1].split("-")[0])
timestamp_end = pd.Timestamp(paths[-1].split("/")[-1].split("-")[0]) + pd.Timedelta("1 hour")
df_epochs_features.index = pd.DatetimeIndex(pd.date_range(timestamp_start, timestamp_end, periods=len(df_epochs_features)))

#%%
for marker, epochs_fun, channels_fun in df_epochs_features.columns:
    ws = pd.Timedelta(window) // pd.Timedelta(f"{chunk_seconds} second")
    if epochs_fun == "mean":
        df_epochs_features[marker][epochs_fun][channels_fun] = df_epochs_features[marker][epochs_fun][channels_fun].rolling(window).mean(raw=True, engine="numba", engine_kwargs={"parallel": True})
    elif epochs_fun == "std":
        df_epochs_features[marker][epochs_fun][channels_fun] = df_epochs_features[marker][epochs_fun][channels_fun].rolling(window).std(raw=True, engine="numba", engine_kwargs={"parallel": True})

df_epochs_features.hist(bins=50, figsize=(20, 20))

df_epochs_features.dropna().to_csv(out_path)
