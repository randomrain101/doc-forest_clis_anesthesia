#%%
#%matplotlib inline
import os
import os.path as op
import openneuro

from mne.datasets import sample
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
dataset = 'ds003754'

bids_root = op.join("../data/raw", dataset)
if not op.isdir(bids_root):
    os.makedirs(bids_root)
    
# Download whole data set (~16GB)
# /sub-0006/ieeg/sub-0006_task-ana2con_ieeg.edf to ds003754-download(error: Access denied)
# /sub-0011/ieeg/sub-0011_task-ana2con_ieeg.edf to ds003754-download(error: Access denied)
# /sub-0030/ieeg/sub-0030_task-ana2con_ieeg.edf to ds003754-download(error: Access denied)
# /sub-0034/ieeg/SEEG.mat (error: Access denied)
# /sub-0037/ieeg/SEEG.mat (error: Access denied)
# /sub-0038/ieeg/SEEG.mat (error: Access denied)
# - with openneuro cli (recommended):
# openneuro-py download --dataset=ds003754
# - with aws cli:
# aws s3 sync --no-sign-request s3://openneuro.org/ds003754 ds003754-download/
# - with openneuro library:
#openneuro.download(dataset=dataset, target_dir=bids_root)

# %%
print_dir_tree(bids_root, max_depth=4)

# %%
print(make_report(bids_root))

# %%
datatype = 'ieeg'
bids_path = BIDSPath(root=bids_root, datatype=datatype)
print(bids_path.match())

# %%
events_paths = bids_path.match("events")
recordings_paths = bids_path.match("ieeg.edf")

#%%
task = 'ana2con'
suffix = 'ieeg'
subject = "0002"

bids_path = BIDSPath(subject=subject, task=task,
                     suffix=suffix, datatype=datatype, root=bids_root)
raw = read_raw_bids(bids_path)

#%%
t_start = pd.Timedelta(raw.annotations[0]["onset"], unit="s")
t_end = pd.Timedelta(raw.annotations[-1]["onset"], unit="s")
#raw = raw.crop(t_start, t_end)

# %%
df = raw.to_data_frame(index="time", time_format="timedelta")

# %%
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

df_scaled = pd.DataFrame(PCA(n_components=5).fit_transform(normalize(df)), index=df.index)

#%%
t_withdrawal = pd.Timedelta(raw.annotations[1]["onset"], unit="s")
t_awake = pd.Timedelta(raw.annotations[2]["onset"], unit="s")

# %%
pd.options.plotting.backend = "matplotlib"
ax = df_scaled.loc[t_start + pd.Timedelta("8min"): t_end].resample("1s").mean().plot(figsize=(15, 5), grid=True, title="pca components")
# pandas transforms time index to ns
ax.axvline(t_withdrawal // pd.Timedelta("1 ns"), color="green", linestyle="dashed")
ax.axvline(t_awake // pd.Timedelta("1 ns"), color="red", linestyle="dashed")
plt.show()

# %%
df_awake = df.loc[t_awake:]
df_awake_pca = pd.DataFrame(PCA(n_components=5).fit_transform(normalize(df_awake)), index=df_awake.index)

# %%
df_awake_pca.resample("1s").mean().plot(figsize=(15, 5), grid=True, title="pca components when awake")

# %%
