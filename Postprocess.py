
"""Postprocess forecasts (statistics, plots, etc.)"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filenames = ["forecasts_5min_BNI.h5", "forecasts_5min_GHI.h5", "forecasts_10min_BNI.h5", "forecasts_10min_GHI.h5"
                , "forecasts_15min_BNI.h5", "forecasts_15min_GHI.h5", "forecasts_20min_BNI.h5", "forecasts_20min_GHI.h5"
                , "forecasts_25min_BNI.h5", "forecasts_25min_GHI.h5", "forecasts_30min_BNI.h5", "forecasts_30min_GHI.h5"]

def summary_stats(target, filenames, baseline="sp"):
    """Compute summary statistics (MAE, MBE, etc.).

    Parameters
    ----------
    target : str {"ghi", "dni"}
        Target variable.
    filenames : list
        List of filenames (relative or absolute) containing forecast time-series.
    baseline : str
        The baseline forecast to compare against.

    Returns
    -------
    df : pandas.DataFrame
        Summary statistics (MAE, MBE, etc.) by horizon, dataset, feature set,
        and model.

    """

    results = []
    for filename in filenames:
        df = pd.read_hdf(filename, "df")

        # essential metadata
        horizon = df["horizon"].values[0]
        #print(df.describe())

        # error metrics
        for dataset, group in df.groupby("dataset"):
            for model in [baseline, "ols", "ridge", "lasso"]:

                # error metrics [W/m^2]
                error = (group["Pdc_{}_actual".format(target)] - group["Pdc_{}_{}".format(target, model)]).values
                mae = np.nanmean(np.abs(error))
                mbe = np.nanmean(error)
                rmse = np.sqrt(np.nanmean(error ** 2))

                # forecast skill [-]:
                #
                #       s = 1 - RMSE_f / RMSE_p
                #
                # where RMSE_f and RMSE_p are the RMSE of the forecast model
                # and reference baseline model, respectively. (smart persistance model)
                rmse_p = np.sqrt(
                    np.mean((group["Pdc_{}_actual".format(target)] - group["Pdc_{}_{}".format(target, baseline)]) ** 2)
                )
                skill = 1.0 - rmse / rmse_p

                results.append(
                    {
                        "dataset": dataset,  # Train/Test
                        "horizon": horizon,  # 5min, 10min, etc.
                        "model": model,
                        "MAE": mae,
                        "MBE": mbe,
                        "RMSE": rmse,
                        "skill": skill,
                        "baseline": baseline,  # the baseline forecast name
                    }
                )

    # return as a DataFrame
    return pd.DataFrame(results)


def summarize(target):
    """Summarize the forecasts, per horizon range."""

    # intra-hour: 5-30min ahead
    df = summary_stats(
        target,
        glob.glob(
            os.path.join(
                "forecasts", "forecasts_*{}.h5".format(target)
            )
        ),
        baseline="sp",
    )
    df.to_hdf("results_{}_intra-hour.h5".format(target), "df", mode="w")
    print("Intra-hour ({}): {}".format(target, df.shape))

"""  # intra-day: 30min to 3h ahead
    df = summary_stats(
        target,
        glob.glob(
            os.path.join(
                "forecasts", "forecasts_intra-day*{}.h5".format(target)
            )
        ),
        baseline="sp",
    )
    df.to_hdf("results_{}_intra-day.h5".format(target), "df", mode="w")
    print("Intra-day ({}): {}".format(target, df.shape)"""


def summary_table(target):
    """Summary table for paper."""

    # results
    df = pd.read_hdf("results_{}_intra-hour.h5".format(target), "df")
    df = df[df["dataset"] == "Test"]

    df.to_csv("summary_table_{}.csv".format(target), index=False, sep=";", decimal=",")

    # generate table
    for model, group in df.groupby(["model"]):
        meta_str = "{:<6}".format(model)
        mae_str = "MAE: {:.3f} std: {:.2f}".format(group.mean()["MAE"], group.std()["MAE"])
        mbe_str = "MBE: {:.3f} std: {:.2f}".format(group.mean()["MBE"], group.std()["MBE"])
        rmse_str = "RMSE: {:.3f} std: {:.2f}".format(group.mean()["RMSE"], group.std()["RMSE"])
        skill_str = "skill: {:.1f} std: {:.2f}".format(group.mean()["skill"] * 100, group.std()["skill"] * 100)
        print("{:<30} && {:<16} & {:<16} & {:<20} & {:<18} \\\\".format(meta_str, mae_str, mbe_str, rmse_str, skill_str))
       

# computes and prints error metrics
target = "BNI"
summarize(target)
summary_table(target)



target = "GHI"
summarize(target)
summary_table(target)

