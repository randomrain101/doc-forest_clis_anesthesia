import pandas as pd

from config import get_reduction_params

def reduce_to_scalars(markers, reduction_functions, subject_info):
    patient, label = subject_info.name
    epochs = list(range(subject_info["start"], subject_info["end"] - 1))
    out = pd.DataFrame()
    for function_dict in reduction_functions:
        epochs_fun = function_dict["epochs_fun"]
        channels_fun = function_dict["channels_fun"]
        reduction_params = get_reduction_params(epochs_fun=epochs_fun,
                                                channels_fun=channels_fun,
                                                epochs=epochs)
        scalars = markers.reduce_to_scalar(reduction_params)
        
        feature_df = pd.DataFrame(
                        scalars,
                        index=markers.keys(),
                        columns=pd.MultiIndex.from_tuples(
                            tuples=[(patient, label, epochs_fun.__name__, channels_fun.__name__)],
                            names=["subject", "label", "epochs_fun", "channels_fun"]
                            )
                        )
        out = out.append(feature_df.T)
    return out

def reduce_to_epochs(markers, reduction_functions, subject_info, window="5min", chunk_seconds=1, center_rolling=False):
    patient, label = subject_info.name
    epochs = list(range(subject_info["start"], subject_info["end"] - 1))
    out_index = pd.TimedeltaIndex(epochs, unit="s") * chunk_seconds
    out = pd.DataFrame(index=out_index)
    for function_dict in reduction_functions:
        epochs_fun = function_dict["epochs_fun"]
        channels_fun = function_dict["channels_fun"]
        reduction_params = get_reduction_params(epochs_fun=epochs_fun,
                                                channels_fun=channels_fun,
                                                epochs=epochs)
        epochs_reduced = markers.reduce_to_epochs(reduction_params)
        df_epochs_features = pd.DataFrame(epochs_reduced)
        df_epochs_features = df_epochs_features.iloc[epochs].set_index(out_index)
        #print(df_epochs_features)
        if epochs_fun.__name__ == "mean":
            df_epochs_features = df_epochs_features.rolling(window, center=center_rolling).mean(raw=True, engine="numba", engine_kwargs={"parallel": True})
        elif epochs_fun.__name__ == "std":
            df_epochs_features = df_epochs_features.rolling(window, center=center_rolling).std(raw=True, engine="numba", engine_kwargs={"parallel": True})
        else:
            df_epochs_features = df_epochs_features.rolling(window, center=center_rolling).apply(epochs_fun, raw=True, engine="numba")
        df_epochs_features.columns = pd.MultiIndex.from_product((df_epochs_features.columns, [epochs_fun.__name__], [channels_fun.__name__]))
        out = out.join(df_epochs_features)
    out.index = pd.MultiIndex.from_product(([patient], [label], out_index))
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["marker", "epochs_fun", "channels_fun"])
    return out

def reduce_to_epochs_clis(markers, reduction_functions, out_index, window=None):
    out = pd.DataFrame(index=out_index)
    for function_dict in reduction_functions:
        channels_fun = function_dict["channels_fun"]
        epochs_fun = function_dict["epochs_fun"]
        epochs_reduced = markers.reduce_to_epochs(get_reduction_params(channels_fun=channels_fun, epochs_fun=None))
        df_epochs_features = pd.DataFrame(epochs_reduced, index=out_index)
        if window:
            df_epochs_features = df_epochs_features.rolling(window).apply(epochs_fun)
        df_epochs_features.columns = pd.MultiIndex.from_product((df_epochs_features.columns, [epochs_fun.__name__], [channels_fun.__name__]))
        out = out.join(df_epochs_features)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["marker", "epochs_fun", "channels_fun"])
    return out