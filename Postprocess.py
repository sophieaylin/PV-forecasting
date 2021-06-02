
"""Postprocess forecasts (statistics, plots, etc.)"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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
                    np.nanmean((group["Pdc_{}_actual".format(target)] - group["Pdc_{}_{}".format(target, baseline)]).values ** 2)
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
                        "RMSE_p": rmse_p,
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


def summary_table(target):
    """Summary table for paper."""

    # results
    df = pd.read_hdf("results_{}_intra-hour.h5".format(target), "df")
    df = df[df["dataset"] == "Test"]

    df.to_csv("summary_table_{}.csv".format(target), index=False, sep=";", decimal=",")
    sumdata = pd.DataFrame()
    sumdata = pd.DataFrame(columns=["model", "MAE", "st_MAE", "MBE", "st_MBE", "RMSE", "st_RMSE", "skill", "st"])

    # generate table
    for model, group in df.groupby(["model"]):
        meta_str = "{:<6}".format(model)
        mae_str = "MAE: {:.3f} std: {:.2f}".format(group.mean()["MAE"], group.std()["MAE"])
        mbe_str = "MBE: {:.3f} std: {:.2f}".format(group.mean()["MBE"], group.std()["MBE"])
        rmse_str = "RMSE: {:.3f} std: {:.2f}".format(group.mean()["RMSE"], group.std()["RMSE"])
        skill_str = "skill: {:.2f} std: {:.2f}".format(group.mean(skipna=True)["skill"] * 100, group.std(skipna=True)["skill"] * 100)
        print("{:<30} && {:<16} & {:<16} & {:<20} & {:<18} \\\\".format(meta_str, mae_str, mbe_str, rmse_str, skill_str))
        newrow = {"model": model, "MAE": group.mean()["MAE"], "st_MAE": group.std()["MAE"]
                     , "MBE": group.mean()["MBE"], "st_MBE": group.std()["MBE"]
                     , "RMSE": group.mean()["RMSE"], "st_RMSE": group.std()["RMSE"]
                     , "skill": group.mean(skipna=True)["skill"] * 100, "st": group.std(skipna=True)["skill"] * 100}
        sumdata = sumdata.append(newrow, ignore_index=True)

    sumdata.to_csv("summary_mean_table_{}.csv".format(target), index=False, sep=";", decimal=",")


# computes and prints error metrics
target = "BNI"
summarize(target)
summary_table(target)
#summarize_sp(target)


target = "GHI"
summarize(target)
summary_table(target)

"""# Plot begin
# forecast30min_60min
# forecast3h_60min
Pact = np.array(group.Pdc_BNI_actual) 
Ppred = np.array(group.Pdc_BNI_ridge)
Psp = np.array(group.Pdc_BNI_sp)
plt.figure(dpi=200)
plt.plot(Pact[2120:2720], label="actual")
plt.plot(Psp[2120:2720], label="naiv Pers.")
plt.plot(Ppred[2120:2720],  label="predicted")
line = plt.gca().lines
plt.setp(line, linewidth=0.5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel("Time Step/5min", size=10)
plt.ylabel("Power/Watt", size=10)
plt.legend(loc='upper right')

# Plot end"""

