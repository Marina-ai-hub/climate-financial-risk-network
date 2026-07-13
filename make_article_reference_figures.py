from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SUMMARY_PATH = (
    BASE_DIR
    / "outputs"
    / "summary"
    / "batch_interbank_sensitivity_uniform_sector_distribution_no_liquidity.csv"
)
REFERENCE_DIR = BASE_DIR / "outputs" / "analysis" / "article_reference_p30"
INTERBANK_DIR = (
    BASE_DIR / "outputs" / "analysis" / "interbank_sensitivity_p30_reference_run"
)
METRICS_DIR = BASE_DIR / "outputs" / "metrics" / "realistic_mechanism"
OUT_DIR = REFERENCE_DIR / "figures_for_article"


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{name}.png"
    pdf_path = OUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def reference_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH)
    df["interbank_liability_scale"] = pd.to_numeric(
        df["interbank_liability_scale"],
        errors="raise",
    )
    stable = (
        df["baseline_stable"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    return df.loc[
        stable
        & np.isclose(df["interbank_liability_scale"], 1.0)
        & df["density_label"].astype(str).eq("p30")
    ].copy()


def add_error_bars(ax, x, y, low, high, **kwargs) -> None:
    yerr = np.vstack([y - low, high - y])
    ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=4, linewidth=1.2, **kwargs)


def fig02_mean_effects() -> None:
    means = pd.read_csv(REFERENCE_DIR / "reference_scenario_mean_ci.csv")
    label_map = {
        "total_company_asset_loss": "Firm external-asset loss",
        "total_company_payment_shortfall": "Firm payment shortfall",
        "total_bank_incoming_shortfall": "Total bank incoming shortfall",
    }
    order = [
        "total_company_asset_loss",
        "total_company_payment_shortfall",
        "total_bank_incoming_shortfall",
    ]
    effects = means.set_index("metric").loc[order].reset_index()
    effects["label"] = effects["metric"].map(label_map)

    fig, ax = plt.subplots(figsize=(9.2, 3.6))

    y_pos = np.arange(len(effects))[::-1]
    x = effects["estimate"].to_numpy(float)
    xerr = np.vstack(
        [
            x - effects["ci_low"].to_numpy(float),
            effects["ci_high"].to_numpy(float) - x,
        ]
    )

    ax.errorbar(
        x,
        y_pos,
        xerr=xerr,
        fmt="o",
        capsize=4,
    )

    ax.set_yticks(y_pos, effects["label"])
    ax.set_xlabel("Mean value with 95% confidence interval")
    ax.set_title("Financial effects after climate shocks")
    ax.grid(axis="x", alpha=0.25)

    for value, ypos in zip(x, y_pos):
        ax.text(
            value + 8,
            ypos + 0.06,
            f"{value:.2f}",
            ha="left",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, "fig02_reference_mean_effects_after_climate_shock")
    save_figure(fig, "figure_average_simulated_financial_effects_climate_shocks")


def fig03_transmission_scatter() -> None:
    df = reference_summary()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(
        df["total_company_payment_shortfall"],
        df["mean_loss_to_capital_ratio"],
        s=14,
        alpha=0.45,
        color="#4B78A8",
        edgecolors="none",
    )
    rho = (
        df["total_company_payment_shortfall"]
        .rank(method="average")
        .corr(df["mean_loss_to_capital_ratio"].rank(method="average"))
    )
    ax.set_xlabel("Firm payment shortfall")
    ax.set_ylabel("Mean bank loss-to-capital ratio")
    ax.set_title(f"Firm losses and bank vulnerability (Spearman rho = {rho:.3f})")
    fig.tight_layout()
    save_figure(fig, "fig03_reference_company_shortfall_bank_vulnerability")


def fig04_exposure_metrics() -> None:
    data = pd.read_csv(
        METRICS_DIR
        / "shock_weight_placebo_feature_metrics_capital_normalized_density_p30_ibx1_0.csv"
    )
    label_map = {
        "aggregate_firm_exposure_to_capital": "Aggregate exposure",
        "uniform_shock_weighted_sector_exposure_to_capital": "Uniform shock-weighted",
        "actual_shock_weighted_sector_exposure_to_capital": "Actual shock-weighted",
    }
    data["label"] = data["feature"].map(label_map)
    order = [
        "Aggregate exposure",
        "Uniform shock-weighted",
        "Actual shock-weighted",
    ]
    data = data.set_index("label").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(data))
    width = 0.34
    ax.bar(x - width / 2, data["test_spearman"], width, label="Spearman")
    ax.bar(x + width / 2, data["test_r2"], width, label="R2")
    ax.set_xticks(x, data["label"], rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test-set score")
    ax.set_title("Exposure diagnostics, reference calibration")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig04_reference_exposure_diagnostics")


def fig05_conditional_correlations() -> None:
    full = pd.read_csv(REFERENCE_DIR / "reference_bank_conditional_correlations_full.csv")
    label_map = {
        "firm_exposure_to_capital": "Total firm exposure",
        "shock_weighted_exposure_to_capital": "Shock-weighted exposure",
    }
    full["label"] = full["x"].map(label_map)
    groups = [
        full.loc[
            full["label"] == "Total firm exposure",
            "spearman_rho",
        ].dropna().to_numpy(),
        full.loc[
            full["label"] == "Shock-weighted exposure",
            "spearman_rho",
        ].dropna().to_numpy(),
    ]

    labels = [
        f"Total firm exposure\nn={len(groups[0])}",
        f"Shock-weighted exposure\nn={len(groups[1])}",
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.boxplot(
        groups,
        labels=labels,
        showmeans=True,
        widths=0.5,
    )

    ax.axhline(0, linestyle="--", color="grey", linewidth=1)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Exposure vulnerability within the same scenario groups")
    ax.grid(axis="y", alpha=0.25)

    for i, values in enumerate(groups, start=1):
        median_value = np.median(values)
        ax.text(
            i,
            median_value,
            f"median={median_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, "fig05_reference_conditional_exposure_correlations")


def fig06_shock_multiplier() -> None:
    data = pd.read_csv(REFERENCE_DIR / "reference_by_shock_multiplier_ci.csv")
    shock_ratio_ci = (
        data[data["metric"].eq("mean_loss_to_capital_ratio")]
        .sort_values("shock_multiplier")
        .copy()
    )
    shock_count_ci = (
        data[data["metric"].eq("shocked_vulnerable_50_banks")]
        .sort_values("shock_multiplier")
        .copy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    axes[0].errorbar(
        shock_ratio_ci["shock_multiplier"],
        shock_ratio_ci["estimate"],
        yerr=[
            shock_ratio_ci["estimate"] - shock_ratio_ci["ci_low"],
            shock_ratio_ci["ci_high"] - shock_ratio_ci["estimate"],
        ],
        fmt="o-",
        capsize=4,
    )

    axes[0].set_xlabel("Climate shock multiplier")
    axes[0].set_ylabel("Mean loss-to-capital ratio")
    axes[0].set_title("Mean bank vulnerability")
    axes[0].grid(alpha=0.25)
    add_panel_label(axes[0], "a)")

    axes[1].errorbar(
        shock_count_ci["shock_multiplier"],
        shock_count_ci["estimate"],
        yerr=[
            shock_count_ci["estimate"] - shock_count_ci["ci_low"],
            shock_count_ci["ci_high"] - shock_count_ci["estimate"],
        ],
        fmt="o-",
        capsize=4,
    )

    axes[1].set_xlabel("Climate shock multiplier")
    axes[1].set_ylabel("Average number of banks")
    axes[1].set_title("Banks above 50% capital threshold")
    axes[1].grid(alpha=0.25)
    add_panel_label(axes[1], "b)")

    fig.tight_layout()
    save_figure(fig, "fig06_reference_shock_multiplier_sensitivity")


def fig07_interbank_amplification() -> None:
    data = pd.read_csv(INTERBANK_DIR / "16_reference_density_article_table.csv")
    data = data[data["interbank_liability_scale"].isin([0.5, 1.0, 1.5])].copy()
    data = data.sort_values("interbank_liability_scale")

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = data["interbank_liability_scale"].to_numpy(float)
    y = data["delta_mean_loss_to_capital_ratio_mean"].to_numpy(float)
    ax.plot(x, y, marker="o", color="#D9843B")
    add_error_bars(
        ax,
        x,
        y,
        data["delta_mean_loss_to_capital_ratio_ci_low"].to_numpy(float),
        data["delta_mean_loss_to_capital_ratio_ci_high"].to_numpy(float),
        color="#1E293B",
    )
    ax.axhline(0, color="#64748B", linewidth=0.8)
    ax.set_xlabel("Interbank-liability scale")
    ax.set_ylabel("Difference vs scale 0")
    ax.set_title("Interbank amplification at density p30")
    fig.tight_layout()
    save_figure(fig, "fig07_reference_interbank_amplification")


def main() -> None:
    fig02_mean_effects()
    fig03_transmission_scatter()
    fig04_exposure_metrics()
    fig05_conditional_correlations()
    fig06_shock_multiplier()
    fig07_interbank_amplification()


if __name__ == "__main__":
    main()
