from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SUMMARY_PATH = (
    BASE_DIR
    / "outputs"
    / "summary"
    / "batch_interbank_sensitivity_uniform_sector_distribution_no_liquidity.csv"
)

DEFAULT_BANK_DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "gnn"
    / "bank_level_tabular_dataset_realistic_mechanism_ibx1_0_p30.csv"
)

DEFAULT_OUT_DIR = BASE_DIR / "outputs" / "analysis" / "article_reference_p30"

TRUE_VALUES = {"true", "1", "yes", "y", "t"}

SCENARIO_MEAN_METRICS = [
    "total_company_asset_loss",
    "total_company_payment_shortfall",
    "firm_to_bank_incoming_shortfall",
    "bank_to_bank_incoming_shortfall",
    "total_bank_incoming_shortfall",
    "total_bank_payment_shortfall",
    "mean_loss_to_capital_ratio",
    "max_loss_to_capital_ratio",
    "shocked_vulnerable_25_banks",
    "shocked_vulnerable_50_banks",
]

SPEARMAN_PAIRS = [
    ("total_company_payment_shortfall", "firm_to_bank_incoming_shortfall"),
    ("total_company_payment_shortfall", "total_bank_incoming_shortfall"),
    ("total_company_asset_loss", "total_company_payment_shortfall"),
    ("total_company_asset_loss", "mean_loss_to_capital_ratio"),
]

SHOCK_METRICS = [
    "total_company_asset_loss",
    "total_company_payment_shortfall",
    "total_bank_incoming_shortfall",
    "mean_loss_to_capital_ratio",
    "shocked_vulnerable_50_banks",
]

BANK_EXPOSURE_METRICS = [
    "firm_exposure_to_capital",
    "shock_weighted_exposure_to_capital",
]


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def filter_reference(
    summary: pd.DataFrame,
    scale: float,
    density_label: str,
) -> pd.DataFrame:
    require_columns(
        summary,
        [
            "baseline_stable",
            "interbank_liability_scale",
            "density_label",
            "seed_id",
        ],
        "summary",
    )
    summary = summary.copy()
    summary["interbank_liability_scale"] = pd.to_numeric(
        summary["interbank_liability_scale"],
        errors="raise",
    )
    if "shock_multiplier" in summary.columns:
        summary["shock_multiplier"] = pd.to_numeric(
            summary["shock_multiplier"],
            errors="raise",
        )

    mask = (
        parse_bool(summary["baseline_stable"])
        & np.isclose(summary["interbank_liability_scale"], float(scale))
        & summary["density_label"].astype(str).eq(str(density_label))
    )
    reference = summary.loc[mask].copy()
    if reference.empty:
        raise ValueError(
            "Reference sample is empty. Check scale, density_label and baseline_stable."
        )
    return reference


def seed_mean_series(
    df: pd.DataFrame,
    value_col: str,
    seed_col: str = "seed_id",
) -> pd.Series:
    values = pd.to_numeric(df[value_col], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    clean = df[[seed_col]].copy()
    clean[value_col] = values
    return clean.dropna().groupby(seed_col, sort=True)[value_col].mean()


def bootstrap_mean_ci(
    seed_means: pd.Series,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = seed_means.to_numpy(dtype=float)
    if len(values) < 2 or n_bootstrap <= 0:
        return np.nan, np.nan
    sampled = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    draws = values[sampled].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_seed_means(
    df: pd.DataFrame,
    metrics: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for metric_index, metric in enumerate(metrics):
        require_columns(df, [metric], "metric")
        seed_means = seed_mean_series(df, metric)
        rng = np.random.default_rng(seed + metric_index)
        ci_low, ci_high = bootstrap_mean_ci(seed_means, n_bootstrap, rng)
        rows.append(
            {
                "metric": metric,
                "estimate": float(seed_means.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_rows": int(len(df)),
                "n_clusters": int(len(seed_means)),
            }
        )
    return pd.DataFrame(rows)


def spearman_corr(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    clean = (
        df[[x_col, y_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    if len(clean) < 5:
        return np.nan
    if clean[x_col].nunique() < 2 or clean[y_col].nunique() < 2:
        return np.nan

    x_rank = clean[x_col].rank(method="average")
    y_rank = clean[y_col].rank(method="average")
    value = x_rank.corr(y_rank, method="pearson")
    return float(value) if pd.notna(value) else np.nan


def bootstrap_spearman_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_bootstrap: int,
    rng: np.random.Generator,
    seed_col: str = "seed_id",
) -> tuple[float, float]:
    seeds = np.array(sorted(df[seed_col].astype(str).unique()))
    if len(seeds) < 2 or n_bootstrap <= 0:
        return np.nan, np.nan

    groups = {
        str(seed_value): group
        for seed_value, group in df.groupby(seed_col, sort=False)
    }
    draws = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        boot = pd.concat([groups[str(seed_value)] for seed_value in sampled])
        draws.append(spearman_corr(boot, x_col, y_col))

    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize_spearman_pairs(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for pair_index, (x_col, y_col) in enumerate(pairs):
        require_columns(df, [x_col, y_col], "spearman pair")
        rng = np.random.default_rng(seed + pair_index)
        ci_low, ci_high = bootstrap_spearman_ci(
            df,
            x_col,
            y_col,
            n_bootstrap,
            rng,
        )
        rows.append(
            {
                "x": x_col,
                "y": y_col,
                "estimate": spearman_corr(df, x_col, y_col),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_rows": int(len(df)),
                "n_clusters": int(df["seed_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_shock_multiplier(
    reference: pd.DataFrame,
    metrics: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for multiplier, group in reference.groupby("shock_multiplier", sort=True):
        table = summarize_seed_means(
            group,
            metrics,
            n_bootstrap=n_bootstrap,
            seed=seed + int(round(float(multiplier) * 100)),
        )
        table.insert(0, "shock_multiplier", float(multiplier))
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def summarize_shock_multiplier_difference(
    reference: pd.DataFrame,
    metrics: list[str],
    low_multiplier: float,
    high_multiplier: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    keys = [
        "seed_id",
        "country",
        "scenario_name",
        "density_label",
        "interbank_liability_scale",
    ]
    require_columns(reference, keys + ["shock_multiplier"], "shock pairing")

    low = reference[np.isclose(reference["shock_multiplier"], low_multiplier)]
    high = reference[np.isclose(reference["shock_multiplier"], high_multiplier)]

    paired = high[keys + metrics].merge(
        low[keys + metrics],
        on=keys,
        suffixes=("_high", "_low"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("No paired shock-multiplier rows found.")

    rows = []
    for metric_index, metric in enumerate(metrics):
        diff_col = f"diff_{metric}"
        paired[diff_col] = paired[f"{metric}_high"] - paired[f"{metric}_low"]
        seed_means = seed_mean_series(paired, diff_col)
        rng = np.random.default_rng(seed + metric_index)
        ci_low, ci_high = bootstrap_mean_ci(seed_means, n_bootstrap, rng)
        rows.append(
            {
                "metric": metric,
                "low_multiplier": float(low_multiplier),
                "high_multiplier": float(high_multiplier),
                "difference_high_vs_low": float(seed_means.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_pairs": int(len(paired)),
                "n_clusters": int(len(seed_means)),
            }
        )
    return pd.DataFrame(rows)


def summarize_bank_conditional_correlations(
    bank_df: pd.DataFrame,
    scale: float,
    density_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        "country",
        "scenario_name",
        "shock_multiplier",
        "density_label",
        "interbank_liability_scale",
        "target_loss_to_capital_ratio",
        *BANK_EXPOSURE_METRICS,
    ]
    require_columns(bank_df, required, "bank-level data")

    bank_df = bank_df.copy()
    bank_df["interbank_liability_scale"] = pd.to_numeric(
        bank_df["interbank_liability_scale"],
        errors="raise",
    )
    mask = (
        bank_df["density_label"].astype(str).eq(str(density_label))
        & np.isclose(bank_df["interbank_liability_scale"], float(scale))
    )
    bank_df = bank_df.loc[mask].copy()
    if bank_df.empty:
        raise ValueError("No bank-level rows remain after reference filtering.")

    group_cols = [
        "country",
        "scenario_name",
        "shock_multiplier",
        "density_label",
        "interbank_liability_scale",
    ]
    rows = []
    for group_key, group in bank_df.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, group_key))
        for exposure_col in BANK_EXPOSURE_METRICS:
            rows.append(
                {
                    **base,
                    "x": exposure_col,
                    "y": "target_loss_to_capital_ratio",
                    "n": int(len(group)),
                    "spearman_rho": spearman_corr(
                        group,
                        exposure_col,
                        "target_loss_to_capital_ratio",
                    ),
                }
            )
    full = pd.DataFrame(rows)
    summary = (
        full.groupby("x", sort=True)["spearman_rho"]
        .agg(
            n_groups="count",
            mean_rho="mean",
            median_rho="median",
            min_rho="min",
            max_rho="max",
        )
        .reset_index()
    )

    wide = full.pivot_table(
        index=group_cols,
        columns="x",
        values="spearman_rho",
    ).reset_index()
    wide["gain"] = (
        wide["shock_weighted_exposure_to_capital"]
        - wide["firm_exposure_to_capital"]
    )
    gain = pd.DataFrame(
        [
            {
                "n_groups": int(len(wide)),
                "mean_gain": float(wide["gain"].mean()),
                "median_gain": float(wide["gain"].median()),
                "min_gain": float(wide["gain"].min()),
                "max_gain": float(wide["gain"].max()),
                "share_groups_weighted_better": float((wide["gain"] > 0).mean()),
            }
        ]
    )
    return full, summary, gain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build article tables for the p30, scale=1.0 reference calibration."
    )
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--bank-data-path", type=Path, default=DEFAULT_BANK_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--density-label", type=str, default="p30")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--spearman-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--low-shock-multiplier", type=float, default=0.75)
    parser.add_argument("--high-shock-multiplier", type=float, default=1.50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_path)
    reference = filter_reference(summary, args.scale, args.density_label)

    metadata = {
        "summary_path": str(args.summary_path),
        "bank_data_path": str(args.bank_data_path),
        "out_dir": str(args.out_dir),
        "reference_scale": float(args.scale),
        "reference_density_label": str(args.density_label),
        "n_reference_rows": int(len(reference)),
        "n_reference_seeds": int(reference["seed_id"].nunique()),
        "n_countries": int(reference["country"].nunique()),
        "n_scenarios": int(reference["scenario_name"].nunique()),
        "shock_multipliers": sorted(
            float(value) for value in reference["shock_multiplier"].unique()
        ),
        "n_bootstrap": int(args.n_bootstrap),
        "spearman_bootstrap": int(args.spearman_bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "estimand": "equal_weight_mean_of_seed_level_means",
    }
    (args.out_dir / "reference_calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    reference[
        [
            "graph_id",
            "seed_id",
            "country",
            "scenario_name",
            "density_label",
            "interbank_liability_scale",
            "shock_multiplier",
            "baseline_stable",
        ]
    ].to_csv(args.out_dir / "reference_rows.csv", index=False)

    summarize_seed_means(
        reference,
        SCENARIO_MEAN_METRICS,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    ).to_csv(args.out_dir / "reference_scenario_mean_ci.csv", index=False)

    summarize_spearman_pairs(
        reference,
        SPEARMAN_PAIRS,
        n_bootstrap=args.spearman_bootstrap,
        seed=args.bootstrap_seed + 1000,
    ).to_csv(args.out_dir / "reference_scenario_spearman_ci.csv", index=False)

    summarize_by_shock_multiplier(
        reference,
        SHOCK_METRICS,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed + 2000,
    ).to_csv(args.out_dir / "reference_by_shock_multiplier_ci.csv", index=False)

    summarize_shock_multiplier_difference(
        reference,
        SHOCK_METRICS,
        low_multiplier=args.low_shock_multiplier,
        high_multiplier=args.high_shock_multiplier,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed + 3000,
    ).to_csv(args.out_dir / "reference_shock_multiplier_difference_ci.csv", index=False)

    if args.bank_data_path.exists():
        bank_df = pd.read_csv(args.bank_data_path)
        full, bank_summary, gain = summarize_bank_conditional_correlations(
            bank_df,
            scale=args.scale,
            density_label=args.density_label,
        )
        full.to_csv(
            args.out_dir / "reference_bank_conditional_correlations_full.csv",
            index=False,
        )
        bank_summary.to_csv(
            args.out_dir / "reference_bank_conditional_correlations_summary.csv",
            index=False,
        )
        gain.to_csv(
            args.out_dir / "reference_bank_conditional_gain_summary.csv",
            index=False,
        )

    print("Reference calibration analysis completed.")
    print(f"Rows: {metadata['n_reference_rows']:,}")
    print(f"Seeds: {metadata['n_reference_seeds']}")
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
