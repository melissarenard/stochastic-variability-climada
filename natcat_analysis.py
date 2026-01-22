"""
Results
"""

import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy import sparse as sp


def read_losses(
    homogeneous: bool,
    n_years: int,
    loc=False,
    p_loc=0.5,
    intensity=False,
    damage=False,
    n_sim=1_000_000,
    dir="Outputs",
) -> npt.NDArray[np.float64]:

    loss_df = pd.read_parquet(
        f"{dir}/{homogeneous}_{n_sim}_{n_years}_{loc}_{p_loc}_{intensity}_{damage}.parquet"
    )

    loss_per_sim = np.array(loss_df.groupby("simulation")["total_loss"].sum())

    n_sim_missing = n_sim - len(loss_per_sim)
    loss_per_sim = np.append(loss_per_sim, [0] * n_sim_missing)

    return loss_per_sim


def read_losses_region(
    homogeneous: bool,
    n_years: int,
    loc=False,
    p_loc=0.5,
    intensity=False,
    damage=False,
    n_sim=1_000_000,
    dir="Outputs",
):
    
    loss_df = pd.read_parquet(
        f"{dir}/{homogeneous}_{n_sim}_{n_years}_{loc}_{p_loc}_{intensity}_{damage}.parquet"
    )

    cols_to_keep = loss_df.columns[loss_df.columns.str.contains("simulation|loss", regex = True)]
    cols_to_keep = [col for col in cols_to_keep if col not in ["total_loss", "larger_losses"]]

    loss_df_regions = loss_df[cols_to_keep]

    loss_per_sim_region = loss_df_regions.groupby("simulation").sum()
    n_sim_missing = n_sim - loss_per_sim_region.shape[0]
    new_rows = pd.DataFrame(0, index = range(n_sim_missing), columns = loss_per_sim_region.columns)
    loss_per_sim_region = pd.concat([loss_per_sim_region, new_rows], ignore_index=True)

    return loss_per_sim_region


def read_cyclone_counts(
        homogeneous: bool,
        n_years: int,
        loc: bool = False,
        p_loc: float = 0.5,
        intensity: bool = False,
        damage: bool = False,
        n_sim: int = 1_000_000,
        dir: str = "Outputs",
):
    """
    For internal use. Used for checking that the number of cyclones simulated is in line with model assumptions. 
    """
    cyclones_df  = pd.read_csv(
        f"{dir}/{homogeneous}_{n_sim}_{n_years}_{loc}_{p_loc}_{intensity}_{damage}_cyclone_counts.csv"
    )

    cyclone_count_per_sim = np.array(cyclones_df.filter(like = "Number of cyclones").sum(1))

    return cyclone_count_per_sim


def plot_return_period_curves(
    losses_per_sim: list, 
    legend_labels: dict, 
    dataset_colors: dict | None = None,
    dataset_linestyles: dict | None = None,
    figsize=(9, 5),
    legend=True,
    fontsize=12,
    n_sim=1_000_000, 
    rp=250,
    savefig=False,
    save_dir=None,
    y_limit=None
):

    XMAX = 250

    exceed_freq = np.linspace(1 / n_sim, 1, n_sim)
    return_period = 1 / exceed_freq[::-1]
    max_idx = int((1 - 1 / rp) * n_sim + 1)

    fig = plt.figure(figsize=figsize)

    AX_WIDTH_IN  = 6.0
    AX_HEIGHT_IN = 4.0

    fig_w, fig_h = fig.get_size_inches()

    ax = fig.add_axes([
        0.1,                                  
        0.15,                                 
        AX_WIDTH_IN / fig_w,                  
        AX_HEIGHT_IN / fig_h                  
    ])

    ymax = 0

    for loss_per_sim in losses_per_sim:
        sorted_values = np.sort(loss_per_sim)
        ymax = max(ymax, sorted_values[:max_idx].max())

        label = None
        for key, value in legend_labels.items():
            if np.array_equal(loss_per_sim, value):
                label = key
                break

        color = dataset_colors.get(id(loss_per_sim)) if dataset_colors else None
        linestyle = dataset_linestyles.get(id(loss_per_sim), "-") if dataset_linestyles else "-"
        

        ax.plot(return_period[:max_idx], sorted_values[:max_idx], label=label, color=color, linestyle=linestyle)

    ax.set_xlim(0, XMAX)
    if y_limit is None:
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(0, y_limit)

    ax.set_xlabel("Return period", fontsize=fontsize)
    ax.set_ylabel("Loss (USD)", fontsize=fontsize)

    ax.tick_params(axis="both", labelsize=fontsize)
    ax.yaxis.get_offset_text().set_fontsize(fontsize)


    if legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            prop={'family': 'monospace', 'size': fontsize + 2}
        )

    if savefig:
        plt.savefig(
            save_dir,
            dpi=300,
            bbox_inches="tight"
        )

    return None


def risk_measures(losses_per_sim: list, column_labels: dict, units: int) -> pd.DataFrame:

    df = pd.DataFrame()

    for loss_per_sim in losses_per_sim:
        mean = loss_per_sim.mean()
        sd = loss_per_sim.std()
        VaR_50 = np.percentile(loss_per_sim, 50)
        VaR_75 = np.percentile(loss_per_sim, 75)
        VaR_95 = np.percentile(loss_per_sim, 95)
        VaR_99 = np.percentile(loss_per_sim, 99)
        VaR_995 = np.percentile(loss_per_sim, 99.5)
        TVaR_99 = np.mean(loss_per_sim[loss_per_sim > VaR_99])
        TVaR_995 = np.mean(loss_per_sim[loss_per_sim > VaR_995])

        all_measures = [
            mean,
            sd,
            VaR_50,
            VaR_75,
            VaR_95,
            VaR_99,
            VaR_995,
            TVaR_99,
            TVaR_995,
        ]

        for key, value in column_labels.items():
            if np.array_equal(loss_per_sim, value):
                colname = key

        df[colname] = all_measures

    df_styled = df.style.format(lambda x: f"{x/units:,.0f}")

    return df_styled
