#%%
import sys
sys.path.append("/home/oliver/doc-forest_clis_ba_ws21/nice")

import os.path as op
import os

import mne

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import base_psd, marker_list
from nice import Markers

from mne_bids import BIDSPath, read_raw_bids, read


def participant_raw_iterator(participants_table: pd.DataFrame,
                             bids_root: str = '../../data/raw/ds003754',
                             task: str = 'ana2con',
                             suffix: str = 'ieeg',
                             datatype: str = "ieeg",
                             skip_con2ana: bool = True) -> mne.io.Raw:
    for i in participants.index:
        subject = participants_table["participant_id"].loc[i].split("-")[-1]
        # ignore con2ana data
        if subject[0] == "1" and skip_con2ana:
            continue
        print(subject)
        bids_path = BIDSPath(subject=subject,
                             task=task,
                             suffix=suffix,
                             datatype=datatype,
                             root=bids_root)
        try:
            raw = read_raw_bids(bids_path)
        except TypeError:
            print(f"subject not found.")
            continue
        yield raw
        
def compute_and_save_markers(markers_list: list,
                             epochs: mne.Epochs,
                             out_path: str = "../../data/processed/seeg_markers") -> Markers:
    mc = Markers(markers_list)
    mc.fit(epochs)    
    subject_name = epochs.info["subject_info"]["his_id"]
    path = out_path + "/" + f"{subject_name}-markers.hdf5"
    mc.save(path, overwrite=True)
    return mc
    
def set_annotations_on_raw(raw: mne.io.Raw, include_waking_period_as_unconscious: bool = False) -> mne.io.Raw:
    descriptions = raw.annotations.description
    
    withdrawal_index = descriptions.tolist().index('Withdrawal of anesthetic')
    waking_up_index = withdrawal_index + 1
    
    if include_waking_period_as_unconscious:
        duration_unconscious = raw.annotations[waking_up_index]["onset"]
    else:
        duration_unconscious = raw.annotations[withdrawal_index]["onset"]
    duration_conscious = raw.times[-1] - raw.annotations[waking_up_index]["onset"]

    # set annotation durations
    annotations = mne.Annotations(
        onset=[raw.times[0], raw.annotations[waking_up_index]["onset"]],
        duration=[duration_unconscious, duration_conscious],
        description=["under_anaesthesia", "awake"],
        orig_time=raw.info["meas_date"]
        )
    raw = raw.set_annotations(annotations)
    
    return raw

def create_epochs_from_annotations(raw: mne.io.Raw, chunk_seconds:int=1) -> mne.Epochs:
    start_of_rec_annot_key = raw.annotations[0]["description"]
    waking_annot_key = raw.annotations[1]["description"]

    # create chunked events from annotations
    event_ids = {
        start_of_rec_annot_key: 0,
        waking_annot_key: 1 # annotaion for patient waking up
    }
    events = mne.events_from_annotations(raw, event_id=event_ids, chunk_duration=chunk_seconds)

    # create epochs (like rolling window)
    epochs = mne.Epochs(raw, events[0], dict(unconscious=0, conscious=1), - chunk_seconds, 0, preload=True)
    return epochs


#%%
dataset = 'ds003754'
bids_root = op.join("../../data/raw", dataset)

markers_folder = "../../data/processed/seeg_markers"
epoch_labels_path = markers_folder + "/" + "epoch_labels.csv"

task = 'ana2con'
datatype = 'ieeg'
suffix = 'ieeg'

include_waking_period_as_unconscious = True

# wether to recompute and overwrite existing markers
overwrite = True

# overwrite existing epoch labels
if overwrite:
    pd.DataFrame().to_csv(epoch_labels_path, header=False)

# load list of participants
participants = pd.read_csv(bids_root + '/participants.tsv', sep='\t', header=0)

#%%
annot_list = []
for raw in participant_raw_iterator(participants_table=participants, bids_root=bids_root):
    # check if markers exist already
    subject_name = raw.info["subject_info"]["his_id"]
    
    path = markers_folder + "/" + f"{subject_name}-markers.hdf5"
    if op.isfile(path) and not overwrite:
        print(f"{path} already exists and will not be overwritten")
        continue
    
    # skip file if annotations are erronous
    if len(raw.annotations) > 4:
        print("raw is missing annotations ... skipping")
        continue
    try:
        raw = set_annotations_on_raw(raw, include_waking_period_as_unconscious);
    except ValueError as e:
        print(e)
        continue
    
    # divide raw into epochs
    from config import chunk_seconds
    epochs = create_epochs_from_annotations(raw, chunk_seconds)
    
    # write label indices to csv
    if op.isfile(epoch_labels_path):
        mode = "a"
        header = False
    else:
        mode = "w"
        header = True
    
    epochs_df = epochs.to_data_frame()
    
    df_epoch_labels = epochs_df.groupby("condition")["epoch"].first().rename("start").to_frame() \
        .join(epochs_df.groupby("condition")["epoch"].last().rename("end").to_frame()) - 1
    df_epoch_labels = pd.concat([df_epoch_labels], keys=[subject_name], names=['patient'])
    df_epoch_labels.to_csv(epoch_labels_path, mode=mode, header=header)
    
    # fit base psd
    base_psd.fit(epochs)
    
    # compute and save marker
    mc = compute_and_save_markers(markers_list=marker_list.copy(), epochs=epochs)
    del mc, epochs, raw

# %%
