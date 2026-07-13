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

DEFAULT_OUT_DIR = (
    BASE_DIR
    / "outputs"
    / "analysis"
    / "interbank_sensitivity"
)

SCALE_COL = "interbank_liability_scale"
CLUSTER_COL = "seed_id"
DENSITY_COL = "density_label"
BASE_SCALE = 0.0

# Values below this tolerance are treated as floating-point noise.
# Keep this at least as large as the numerical tolerance used by the clearing solver.
NUMERICAL_ZERO_TOL = 1e-10

# Bootstrap intervals are suppressed when too few independent seed clusters remain.
MIN_CLUSTERS_FOR_CI = 10

# Backward-compatible alias used in comparisons throughout the script.
EPS = NUMERICAL_ZERO_TOL

# These names are intentionally kept consistent with the existing simulation code.
METRICS = [
    "total_bank_incoming_shortfall",
    "total_bank_payment_shortfall",
    "mean_loss_to_capital_ratio",
    "median_loss_to_capital_ratio",
    "max_loss_to_capital_ratio",
    "shocked_vulnerable_25_banks",
    "shocked_vulnerable_50_banks",
    "shocked_bank_capital_breaches",
    "firm_to_bank_incoming_shortfall",
    "bank_to_bank_incoming_shortfall",
    "bank_to_bank_shortfall_share",
]

BASELINE_DIAGNOSTIC_COLS = [
    "baseline_company_defaults",
    "baseline_bank_payment_defaults",
    "baseline_bank_capital_breaches",
    "baseline_failed_banks",
]

# A smaller set used in compact article-oriented output tables.
ARTICLE_EFFECT_METRICS = [
    "total_bank_incoming_shortfall",
    "mean_loss_to_capital_ratio",
    "shocked_vulnerable_50_banks",
    "shocked_bank_capital_breaches",
]

ARTICLE_LEVEL_METRICS = [
    "firm_to_bank_incoming_shortfall",
    "bank_to_bank_incoming_shortfall",
    "bank_to_bank_shortfall_share",
    "mean_loss_to_capital_ratio",
]


def parse_bool(series: pd.Series) -> pd.Series:
    """Parse a CSV boolean column safely."""
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}

    text = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(text.unique()) - true_values - false_values)
    if unknown:
        raise ValueError(
            "Cannot parse baseline_stable as boolean. "
            f"Unexpected values: {unknown}"
        )

    return text.isin(true_values)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def clean_numerical_zero(
    series: pd.Series,
    tolerance: float = NUMERICAL_ZERO_TOL,
) -> pd.Series:
    """Convert tiny floating-point residuals to exact zero."""
    values = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return values.mask(values.abs() <= tolerance, 0.0)


def equal_cluster_mean(
    data: pd.DataFrame,
    value_col: str,
    cluster_col: str = CLUSTER_COL,
) -> float:
    """Mean of within-cluster means, giving every independent seed equal weight."""
    clean = data[[cluster_col, value_col]].copy()
    clean[value_col] = clean_numerical_zero(clean[value_col])
    clean = clean.dropna()
    if clean.empty:
        return np.nan
    return float(clean.groupby(cluster_col, sort=True)[value_col].mean().mean())


def inference_status(n_clusters: int) -> str:
    """Label whether the bootstrap interval is suitable for main inference."""
    if n_clusters >= MIN_CLUSTERS_FOR_CI:
        return "main_inference"
    if n_clusters >= 2:
        return "exploratory_too_few_clusters"
    return "descriptive_only"


def optional_design_columns(df: pd.DataFrame) -> list[str]:
    """Return design columns that should be preserved if present."""
    candidates = [
        "sector_distribution",
        "liquidity_label",
        "include_bank_liquidity",
        "shock_spec_label",
    ]
    return [column for column in candidates if column in df.columns]


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Combined interbank sensitivity summary not found: {path}\n"
            "Run run_single_scenario.py first, or pass --summary-path."
        )

    df = pd.read_csv(path)

    required = [
        "graph_id",
        CLUSTER_COL,
        "country",
        "scenario_name",
        "shock_multiplier",
        DENSITY_COL,
        SCALE_COL,
        "baseline_stable",
        *BASELINE_DIAGNOSTIC_COLS,
        *METRICS,
    ]
    require_columns(df, required, "required analysis")

    df = df.copy()
    df["baseline_stable"] = parse_bool(df["baseline_stable"])
    df[SCALE_COL] = pd.to_numeric(df[SCALE_COL], errors="raise").astype(float)
    df["shock_multiplier"] = pd.to_numeric(
        df["shock_multiplier"], errors="raise"
    ).astype(float)

    for column in BASELINE_DIAGNOSTIC_COLS + METRICS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[METRICS].isna().any().any():
        bad_columns = df[METRICS].columns[df[METRICS].isna().any()].tolist()
        raise ValueError(
            "Analysis metrics contain missing or non-numeric values: "
            f"{bad_columns}"
        )

    pair_keys = [
        CLUSTER_COL,
        "country",
        "scenario_name",
        "shock_multiplier",
        DENSITY_COL,
        *optional_design_columns(df),
        SCALE_COL,
    ]

    duplicated = df.duplicated(pair_keys, keep=False)
    if duplicated.any():
        example = df.loc[duplicated, pair_keys].head(10)
        raise ValueError(
            "The combined summary contains duplicate scenario rows for the same "
            "seed/country/scenario/multiplier/density/scale. Examples:\n"
            f"{example.to_string(index=False)}"
        )

    if not np.isclose(df[SCALE_COL], BASE_SCALE).any():
        raise ValueError(
            f"No scale={BASE_SCALE} rows found. A no-interbank reference is required."
        )

    return df


def baseline_config_columns(df: pd.DataFrame) -> list[str]:
    """
    Baseline clearing is repeated across countries, scenarios and shock multipliers,
    but it is determined by the synthetic network configuration only.
    """
    return [
        CLUSTER_COL,
        DENSITY_COL,
        SCALE_COL,
        *optional_design_columns(df),
    ]


def build_baseline_config_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated scenario rows to one row per baseline configuration."""
    config_cols = baseline_config_columns(df)
    check_cols = ["baseline_stable", *BASELINE_DIAGNOSTIC_COLS]

    grouped = df.groupby(config_cols, dropna=False, sort=True)

    # The same pre-shock baseline is repeated for every country/scenario/multiplier.
    # It must therefore be identical within a baseline configuration.
    for column in check_cols:
        nunique = grouped[column].nunique(dropna=False)
        inconsistent = nunique[nunique > 1]
        if not inconsistent.empty:
            raise ValueError(
                f"Baseline column '{column}' is not constant within some network "
                "configurations. This means baseline results were not reused "
                "consistently across scenario rows."
            )

    configs = grouped[check_cols].first().reset_index()

    configs["excluded_firm_default"] = (
        configs["baseline_company_defaults"] > 0
    )
    configs["excluded_bank_payment_default"] = (
        configs["baseline_bank_payment_defaults"] > 0
    )
    configs["excluded_capital_breach"] = (
        configs["baseline_bank_capital_breaches"] > 0
    )
    configs["excluded_failed_bank"] = configs["baseline_failed_banks"] > 0
    configs["excluded_any_reason"] = ~configs["baseline_stable"]

    implied_stable = ~(
        configs[
            [
                "excluded_firm_default",
                "excluded_bank_payment_default",
                "excluded_capital_breach",
                "excluded_failed_bank",
            ]
        ].any(axis=1)
    )

    if not (implied_stable == configs["baseline_stable"]).all():
        mismatch = configs.loc[
            implied_stable != configs["baseline_stable"],
            config_cols + check_cols,
        ].head(10)
        raise ValueError(
            "baseline_stable is inconsistent with the four baseline exclusion "
            "criteria. Examples:\n"
            f"{mismatch.to_string(index=False)}"
        )

    return configs


def retention_summary(
    configs: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    summary = (
        configs.groupby(group_cols, dropna=False, sort=True)
        .agg(
            candidate_configurations=(CLUSTER_COL, "size"),
            retained_configurations=("baseline_stable", "sum"),
            retention_rate=("baseline_stable", "mean"),
            excluded_configurations=("excluded_any_reason", "sum"),
            excluded_firm_default=("excluded_firm_default", "sum"),
            excluded_bank_payment_default=(
                "excluded_bank_payment_default",
                "sum",
            ),
            excluded_capital_breach=("excluded_capital_breach", "sum"),
            excluded_failed_bank=("excluded_failed_bank", "sum"),
            n_seeds=(CLUSTER_COL, "nunique"),
        )
        .reset_index()
    )

    summary["retention_rate"] = summary["retention_rate"].astype(float)
    return summary


def pair_key_columns(df: pd.DataFrame) -> list[str]:
    return [
        CLUSTER_COL,
        "country",
        "scenario_name",
        "shock_multiplier",
        DENSITY_COL,
        *optional_design_columns(df),
    ]


def build_scale0_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair every positive interbank-liability scale with scale 0 while holding
    seed, country, scenario, shock multiplier and density fixed.
    """
    keys = pair_key_columns(df)

    scale0 = df[np.isclose(df[SCALE_COL], BASE_SCALE)].copy()
    positive = df[df[SCALE_COL] > BASE_SCALE + EPS].copy()

    if positive.empty:
        raise ValueError("No positive interbank-liability scales found.")

    scale0_cols = keys + [
        "graph_id",
        "baseline_stable",
        *METRICS,
    ]
    scale0 = scale0[scale0_cols].rename(
        columns={
            "graph_id": "graph_id_scale0",
            "baseline_stable": "baseline_stable_scale0",
            **{metric: f"{metric}_scale0" for metric in METRICS},
        }
    )

    positive = positive.rename(
        columns={
            "graph_id": "graph_id_current",
            "baseline_stable": "baseline_stable_current",
        }
    )

    paired = positive.merge(
        scale0,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = paired["_merge"] != "both"
    if unmatched.any():
        example = paired.loc[unmatched, keys + [SCALE_COL]].head(10)
        raise ValueError(
            "Some positive-scale rows do not have a matching scale-0 row. "
            "Examples:\n"
            f"{example.to_string(index=False)}"
        )

    paired = paired.drop(columns="_merge")
    paired["common_support"] = (
        paired["baseline_stable_current"]
        & paired["baseline_stable_scale0"]
    )

    for metric in METRICS:
        current = clean_numerical_zero(paired[metric])
        reference = clean_numerical_zero(paired[f"{metric}_scale0"])
        delta = clean_numerical_zero(current - reference)

        paired[f"delta_{metric}_vs_scale0"] = delta
        paired[f"relative_delta_{metric}_vs_scale0"] = np.where(
            reference.abs() > NUMERICAL_ZERO_TOL,
            delta / reference.abs(),
            np.nan,
        )

    return paired


def cluster_bootstrap_mean(
    data: pd.DataFrame,
    value_col: str,
    rng: np.random.Generator,
    n_boot: int,
    cluster_col: str = CLUSTER_COL,
) -> tuple[float, float, float, int]:
    """
    Percentile cluster bootstrap for an equal-seed mean.

    First, a mean is calculated within each independent synthetic seed. The
    reported point estimate is then the arithmetic mean of those seed-level
    means, so every seed receives the same weight even when baseline filtering
    leaves a different number of density configurations for different seeds.
    Bootstrap resampling is performed over the seed-level means.
    """
    clean = data[[cluster_col, value_col]].copy()
    clean[value_col] = clean_numerical_zero(clean[value_col])
    clean = clean.dropna()

    if clean.empty:
        return np.nan, np.nan, np.nan, 0

    seed_means = (
        clean.groupby(cluster_col, sort=True)[value_col]
        .mean()
        .to_numpy(dtype=float)
    )
    n_clusters = len(seed_means)
    point = float(seed_means.mean())

    if (
        n_clusters < MIN_CLUSTERS_FOR_CI
        or n_boot <= 0
    ):
        return point, np.nan, np.nan, n_clusters

    sampled_indices = rng.integers(
        0,
        n_clusters,
        size=(n_boot, n_clusters),
    )
    boot_estimates = seed_means[sampled_indices].mean(axis=1)

    ci_low, ci_high = np.quantile(boot_estimates, [0.025, 0.975])
    return point, float(ci_low), float(ci_high), n_clusters


def count_unique_configurations(
    data: pd.DataFrame,
    include_scale: bool = False,
) -> int:
    columns = [CLUSTER_COL, DENSITY_COL, *optional_design_columns(data)]
    if include_scale:
        columns.append(SCALE_COL)
    return int(data[columns].drop_duplicates().shape[0])


def summarize_paired_effects(
    common_support: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
    n_boot: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Create long-format paired effect summaries with seed-cluster intervals."""
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict] = []

    grouped = common_support.groupby(group_cols, dropna=False, sort=True)

    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_info = dict(zip(group_cols, group_values))

        for metric in metrics:
            delta_col = f"delta_{metric}_vs_scale0"
            estimate, ci_low, ci_high, n_clusters = cluster_bootstrap_mean(
                group,
                value_col=delta_col,
                rng=rng,
                n_boot=n_boot,
            )

            delta = clean_numerical_zero(group[delta_col]).dropna()

            row = {
                **group_info,
                "metric": metric,
                "n_paired_runs": int(len(group)),
                "n_seed_clusters": int(n_clusters),
                "n_common_support_configurations": count_unique_configurations(
                    group
                ),
                "mean_scale0": equal_cluster_mean(group, f"{metric}_scale0"),
                "mean_current_scale": equal_cluster_mean(group, metric),
                "mean_difference_vs_scale0": estimate,
                "cluster_bootstrap_ci_low": ci_low,
                "cluster_bootstrap_ci_high": ci_high,
                "estimand": "mean_of_seed_level_means",
                "inference_status": inference_status(n_clusters),
                "ci_available": bool(np.isfinite(ci_low) and np.isfinite(ci_high)),
                "median_difference_vs_scale0": float(delta.median()),
                "p05_difference_vs_scale0": float(delta.quantile(0.05)),
                "p95_difference_vs_scale0": float(delta.quantile(0.95)),
                "share_positive_difference": float((delta > 0.0).mean()),
                "share_negative_difference": float((delta < 0.0).mean()),
                "share_numerically_zero_difference": float(
                    (delta == 0.0).mean()
                ),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_levels_on_common_support(
    common_support: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
    n_boot: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Summarize current-scale levels on the same common-support sample."""
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict] = []

    grouped = common_support.groupby(group_cols, dropna=False, sort=True)

    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_info = dict(zip(group_cols, group_values))

        for metric in metrics:
            estimate, ci_low, ci_high, n_clusters = cluster_bootstrap_mean(
                group,
                value_col=metric,
                rng=rng,
                n_boot=n_boot,
            )
            values = clean_numerical_zero(group[metric]).dropna()

            rows.append(
                {
                    **group_info,
                    "metric": metric,
                    "n_runs": int(len(group)),
                    "n_seed_clusters": int(n_clusters),
                    "n_common_support_configurations": count_unique_configurations(
                        group
                    ),
                    "mean": estimate,
                    "cluster_bootstrap_ci_low": ci_low,
                    "cluster_bootstrap_ci_high": ci_high,
                    "estimand": "mean_of_seed_level_means",
                    "inference_status": inference_status(n_clusters),
                    "ci_available": bool(np.isfinite(ci_low) and np.isfinite(ci_high)),
                    "median": float(values.median()),
                    "p90": float(values.quantile(0.90)),
                    "p95": float(values.quantile(0.95)),
                    "maximum": float(values.max()),
                    "nonzero_share": float((values != 0.0).mean()),
                }
            )

    return pd.DataFrame(rows)


def build_distribution_summary(common_support: pd.DataFrame) -> pd.DataFrame:
    """Distributional diagnostics needed for cautious density interpretation."""
    rows: list[dict] = []

    for (scale, density), group in common_support.groupby(
        [SCALE_COL, DENSITY_COL],
        dropna=False,
        sort=True,
    ):
        b2b = clean_numerical_zero(group["bank_to_bank_incoming_shortfall"])
        f2b = clean_numerical_zero(group["firm_to_bank_incoming_shortfall"])
        share = clean_numerical_zero(group["bank_to_bank_shortfall_share"])

        rows.append(
            {
                SCALE_COL: float(scale),
                DENSITY_COL: density,
                "n_runs": int(len(group)),
                "n_seeds": int(group[CLUSTER_COL].nunique()),
                "n_common_support_configurations": count_unique_configurations(
                    group
                ),
                "mean_firm_to_bank_shortfall": float(f2b.mean()),
                "mean_bank_to_bank_shortfall": float(b2b.mean()),
                "median_bank_to_bank_shortfall": float(b2b.median()),
                "p90_bank_to_bank_shortfall": float(b2b.quantile(0.90)),
                "p95_bank_to_bank_shortfall": float(b2b.quantile(0.95)),
                "max_bank_to_bank_shortfall": float(b2b.max()),
                "nonzero_bank_to_bank_shortfall_share": float(
                    (b2b != 0.0).mean()
                ),
                "mean_bank_to_bank_shortfall_share": float(share.mean()),
                "median_bank_to_bank_shortfall_share": float(share.median()),
            }
        )

    return pd.DataFrame(rows)


def build_density_p50_vs_p05_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair density p50 with p05 within each scale.

    Important: the firm-bank layer and bank characteristics are held fixed by
    seed, but the interbank topology is regenerated for each density setting.
    This is therefore a comparison of density settings, not a nested-edge causal
    experiment.
    """
    required_densities = {"p05", "p50"}
    present = set(df[DENSITY_COL].astype(str).unique())
    if not required_densities.issubset(present):
        return pd.DataFrame()

    keys = [
        CLUSTER_COL,
        "country",
        "scenario_name",
        "shock_multiplier",
        SCALE_COL,
        *optional_design_columns(df),
    ]

    low = df[df[DENSITY_COL] == "p05"].copy()
    high = df[df[DENSITY_COL] == "p50"].copy()

    low_cols = keys + ["baseline_stable", *METRICS]
    low = low[low_cols].rename(
        columns={
            "baseline_stable": "baseline_stable_p05",
            **{metric: f"{metric}_p05" for metric in METRICS},
        }
    )
    high = high.rename(columns={"baseline_stable": "baseline_stable_p50"})

    paired = high.merge(
        low,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    paired["common_support"] = (
        paired["baseline_stable_p50"] & paired["baseline_stable_p05"]
    )

    for metric in METRICS:
        paired[f"delta_{metric}_p50_minus_p05"] = clean_numerical_zero(
            paired[metric] - paired[f"{metric}_p05"]
        )

    return paired


def summarize_density_effects(
    density_pairs: pd.DataFrame,
    n_boot: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if density_pairs.empty:
        return pd.DataFrame()

    common = density_pairs[density_pairs["common_support"]].copy()
    if common.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict] = []

    for scale, group in common.groupby(SCALE_COL, sort=True):
        for metric in METRICS:
            value_col = f"delta_{metric}_p50_minus_p05"
            estimate, ci_low, ci_high, n_clusters = cluster_bootstrap_mean(
                group,
                value_col=value_col,
                rng=rng,
                n_boot=n_boot,
            )
            values = clean_numerical_zero(group[value_col]).dropna()

            rows.append(
                {
                    SCALE_COL: float(scale),
                    "metric": metric,
                    "comparison": "p50_minus_p05",
                    "n_paired_runs": int(len(group)),
                    "n_seed_clusters": int(n_clusters),
                    "mean_difference": estimate,
                    "cluster_bootstrap_ci_low": ci_low,
                    "cluster_bootstrap_ci_high": ci_high,
                    "estimand": "mean_of_seed_level_means",
                    "inference_status": inference_status(n_clusters),
                    "ci_available": bool(np.isfinite(ci_low) and np.isfinite(ci_high)),
                    "density_design": "independent_topology_draws_not_nested",
                    "median_difference": float(values.median()),
                    "share_positive_difference": float((values > 0.0).mean()),
                    "share_negative_difference": float((values < 0.0).mean()),
                }
            )

    return pd.DataFrame(rows)


def build_article_table(
    retention_by_scale: pd.DataFrame,
    effects_by_scale: pd.DataFrame,
    levels_by_scale: pd.DataFrame,
    distribution: pd.DataFrame,
) -> pd.DataFrame:
    """Create one compact row per positive interbank-liability scale."""
    positive_scales = sorted(
        effects_by_scale.loc[
            effects_by_scale[SCALE_COL] > BASE_SCALE + EPS,
            SCALE_COL,
        ].unique()
    )

    rows: list[dict] = []

    for scale in positive_scales:
        row: dict = {SCALE_COL: float(scale)}

        retention_row = retention_by_scale[
            np.isclose(retention_by_scale[SCALE_COL], scale)
        ]
        if not retention_row.empty:
            r = retention_row.iloc[0]
            row.update(
                {
                    "candidate_configurations": int(
                        r["candidate_configurations"]
                    ),
                    "retained_configurations": int(
                        r["retained_configurations"]
                    ),
                    "retention_rate": float(r["retention_rate"]),
                }
            )

        scale_effects = effects_by_scale[
            np.isclose(effects_by_scale[SCALE_COL], scale)
        ]
        for metric in ARTICLE_EFFECT_METRICS:
            selected = scale_effects[scale_effects["metric"] == metric]
            if selected.empty:
                continue
            value = selected.iloc[0]
            prefix = f"delta_{metric}"
            row[f"{prefix}_mean"] = float(value["mean_difference_vs_scale0"])
            row[f"{prefix}_ci_low"] = float(
                value["cluster_bootstrap_ci_low"]
            )
            row[f"{prefix}_ci_high"] = float(
                value["cluster_bootstrap_ci_high"]
            )
            row["n_common_support_configurations"] = int(
                value["n_common_support_configurations"]
            )
            row["n_common_support_runs"] = int(value["n_paired_runs"])
            row["n_seed_clusters"] = int(value["n_seed_clusters"])
            row["inference_status"] = str(value["inference_status"])
            row["estimand"] = str(value["estimand"])

        scale_levels = levels_by_scale[
            np.isclose(levels_by_scale[SCALE_COL], scale)
        ]
        for metric in ARTICLE_LEVEL_METRICS:
            selected = scale_levels[scale_levels["metric"] == metric]
            if selected.empty:
                continue
            value = selected.iloc[0]
            prefix = f"level_{metric}"
            row[f"{prefix}_mean"] = float(value["mean"])
            row[f"{prefix}_ci_low"] = float(
                value["cluster_bootstrap_ci_low"]
            )
            row[f"{prefix}_ci_high"] = float(
                value["cluster_bootstrap_ci_high"]
            )
            row[f"{prefix}_median"] = float(value["median"])
            row[f"{prefix}_p95"] = float(value["p95"])
            row[f"{prefix}_nonzero_share"] = float(value["nonzero_share"])

        scale_distribution = distribution[
            np.isclose(distribution[SCALE_COL], scale)
        ]
        if not scale_distribution.empty:
            # Aggregate the already computed density-specific diagnostics only
            # for a compact descriptive reference. Main inference remains in the
            # paired and level tables above.
            row["mean_of_density_medians_bank_to_bank_shortfall"] = float(
                scale_distribution["median_bank_to_bank_shortfall"].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def write_report(
    path: Path,
    input_path: Path,
    df: pd.DataFrame,
    configs: pd.DataFrame,
    common_support: pd.DataFrame,
    n_boot: int,
    output_files: list[str],
) -> None:
    scales = sorted(df[SCALE_COL].unique().tolist())
    densities = sorted(df[DENSITY_COL].astype(str).unique().tolist())

    lines = [
        "# Interbank sensitivity analysis report",
        "",
        f"Input summary: `{input_path}`",
        f"Scenario-level rows: {len(df):,}",
        f"Independent synthetic seeds: {df[CLUSTER_COL].nunique()}",
        f"Unique baseline configurations: {len(configs):,}",
        f"Interbank-liability scales: {scales}",
        f"Density labels: {densities}",
        f"Cluster-bootstrap draws: {n_boot}",
        f"Numerical-zero tolerance: {NUMERICAL_ZERO_TOL}",
        f"Minimum seed clusters for a reported interval: {MIN_CLUSTERS_FOR_CI}",
        "Estimand: equal-weight mean of seed-level means",
        "",
        "## What the analysis does",
        "",
        "1. Collapses repeated scenario rows to one pre-shock baseline configuration per seed, density and interbank-liability scale.",
        "2. Reports retention after the baseline-stability filter and the reasons for exclusion.",
        "3. Pairs each positive interbank-liability scale with scale 0 using the same seed, country, scenario, shock multiplier and density.",
        "4. Keeps only common-support pairs for which both the positive-scale and scale-0 configurations are baseline-stable.",
        "5. Computes paired differences as equal-weight means of seed-level means and 95% seed-cluster bootstrap intervals.",
        "6. Reports the distribution of the bank-to-bank shortfall by scale and density.",
        "",
        f"Common-support positive-scale scenario pairs: {len(common_support):,}",
        f"Seeds represented on common support: {common_support[CLUSTER_COL].nunique() if not common_support.empty else 0}",
        "",
        "## Interpretation",
        "",
        "- Retention tables show whether high interbank scales or particular densities are disproportionately removed by the baseline-stability filter.",
        "- Paired effect tables estimate the change relative to scale 0 on the same common-support configurations.",
        "- Bootstrap intervals are reported only when at least the minimum number of independent seed clusters remains.",
        "- When an interval is available and includes zero, the direction is not consistently supported across synthetic seeds.",
        "- Even when an interval excludes zero, the effect size should still be assessed for practical importance.",
        "- The p50-versus-p05 output is descriptive only. Interbank topology is regenerated at each density, so it is not a nested-edge causal experiment.",
        "- Tiny residuals at or below the numerical-zero tolerance are set to exact zero before summaries and bootstrap calculations.",
        "",
        "## Output files",
        "",
        *[f"- `{name}`" for name in output_files],
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse interbank-liability scale and density effects using "
            "baseline retention, common support and seed-cluster bootstrap."
        )
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to the combined summary produced by run_single_scenario.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Number of seed-cluster bootstrap draws. Use 0 to skip intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility.",
    )
    parser.add_argument(
        "--reference-density",
        type=str,
        default="p30",
        help=(
            "Density used for the primary article-oriented scale table. "
            "All-density outputs are still produced for diagnostics."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.summary_path}")
    df = load_summary(args.summary_path)
    print(f"Loaded rows: {len(df):,}")
    print(f"Seeds: {df[CLUSTER_COL].nunique()}")
    print(f"Scales: {sorted(df[SCALE_COL].unique())}")
    print(f"Densities: {sorted(df[DENSITY_COL].astype(str).unique())}")

    # ------------------------------------------------------------------
    # 1. Baseline-stability retention
    # ------------------------------------------------------------------
    configs = build_baseline_config_table(df)
    retention_by_scale_density = retention_summary(
        configs,
        [SCALE_COL, DENSITY_COL],
    )
    retention_by_scale = retention_summary(configs, [SCALE_COL])

    # ------------------------------------------------------------------
    # 2. Paired scale effects on common support
    # ------------------------------------------------------------------
    all_pairs = build_scale0_pairs(df)
    common_support = all_pairs[all_pairs["common_support"]].copy()

    if common_support.empty:
        raise ValueError(
            "No common-support pairs remain after requiring baseline stability "
            "at both the positive scale and scale 0."
        )

    effects_by_scale = summarize_paired_effects(
        common_support,
        group_cols=[SCALE_COL],
        metrics=METRICS,
        n_boot=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    effects_by_scale_density = summarize_paired_effects(
        common_support,
        group_cols=[SCALE_COL, DENSITY_COL],
        metrics=METRICS,
        n_boot=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed + 1,
    )

    # ------------------------------------------------------------------
    # 3. Channel levels and distribution on the same common-support sample
    # ------------------------------------------------------------------
    levels_by_scale = summarize_levels_on_common_support(
        common_support,
        group_cols=[SCALE_COL],
        metrics=ARTICLE_LEVEL_METRICS,
        n_boot=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed + 2,
    )
    levels_by_scale_density = summarize_levels_on_common_support(
        common_support,
        group_cols=[SCALE_COL, DENSITY_COL],
        metrics=ARTICLE_LEVEL_METRICS,
        n_boot=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed + 3,
    )
    distribution_by_scale_density = build_distribution_summary(common_support)

    # ------------------------------------------------------------------
    # 4. Density-setting comparison p50 versus p05
    # ------------------------------------------------------------------
    density_pairs = build_density_p50_vs_p05_pairs(df)
    density_effects = summarize_density_effects(
        density_pairs,
        n_boot=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed + 4,
    )

    # ------------------------------------------------------------------
    # 5. Compact table for the article
    # ------------------------------------------------------------------
    article_table = build_article_table(
        retention_by_scale=retention_by_scale,
        effects_by_scale=effects_by_scale,
        levels_by_scale=levels_by_scale,
        distribution=distribution_by_scale_density,
    )

    # Primary article-oriented table at one pre-specified reference density.
    # This avoids mixing different interbank topologies in the main scale result.
    reference_density = str(args.reference_density)
    if reference_density not in set(df[DENSITY_COL].astype(str).unique()):
        raise ValueError(
            f"Reference density {reference_density!r} is not present in the data."
        )

    reference_article_table = build_article_table(
        retention_by_scale=retention_by_scale_density.loc[
            retention_by_scale_density[DENSITY_COL].astype(str)
            == reference_density
        ].copy(),
        effects_by_scale=effects_by_scale_density.loc[
            effects_by_scale_density[DENSITY_COL].astype(str)
            == reference_density
        ].copy(),
        levels_by_scale=levels_by_scale_density.loc[
            levels_by_scale_density[DENSITY_COL].astype(str)
            == reference_density
        ].copy(),
        distribution=distribution_by_scale_density.loc[
            distribution_by_scale_density[DENSITY_COL].astype(str)
            == reference_density
        ].copy(),
    )
    reference_article_table.insert(1, DENSITY_COL, reference_density)

    outputs: dict[str, pd.DataFrame] = {
        "01_baseline_configurations.csv": configs,
        "02_retention_by_scale_density.csv": retention_by_scale_density,
        "03_retention_by_scale.csv": retention_by_scale,
        "04_all_scale0_pairs.csv": all_pairs,
        "05_common_support_pairs.csv": common_support,
        "06_paired_effects_by_scale.csv": effects_by_scale,
        "07_paired_effects_by_scale_density.csv": effects_by_scale_density,
        "08_channel_levels_by_scale.csv": levels_by_scale,
        "09_channel_levels_by_scale_density.csv": levels_by_scale_density,
        "10_bank_to_bank_distribution_by_scale_density.csv": (
            distribution_by_scale_density
        ),
        "11_density_p50_vs_p05_pairs.csv": density_pairs,
        "12_density_p50_vs_p05_effects.csv": density_effects,
        "13_article_interbank_table.csv": article_table,
        "16_reference_density_article_table.csv": reference_article_table,
    }

    for filename, table in outputs.items():
        path = args.out_dir / filename
        table.to_csv(path, index=False)
        print(f"Saved: {path}")

    metadata = {
        "summary_path": str(args.summary_path),
        "out_dir": str(args.out_dir),
        "n_rows": int(len(df)),
        "n_seeds": int(df[CLUSTER_COL].nunique()),
        "n_unique_baseline_configurations": int(len(configs)),
        "n_common_support_pairs": int(len(common_support)),
        "interbank_liability_scales": sorted(
            float(value) for value in df[SCALE_COL].unique()
        ),
        "density_labels": sorted(df[DENSITY_COL].astype(str).unique().tolist()),
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "numerical_zero_tolerance": float(NUMERICAL_ZERO_TOL),
        "minimum_clusters_for_ci": int(MIN_CLUSTERS_FOR_CI),
        "estimand": "equal_weight_mean_of_seed_level_means",
        "density_comparison_design": "independent_topology_draws_not_nested",
        "reference_density_for_primary_article_table": reference_density,
        "metrics": METRICS,
    }
    metadata_path = args.out_dir / "14_analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {metadata_path}")

    report_path = args.out_dir / "15_READ_ME_FIRST.md"
    write_report(
        path=report_path,
        input_path=args.summary_path,
        df=df,
        configs=configs,
        common_support=common_support,
        n_boot=args.n_bootstrap,
        output_files=[*outputs.keys(), metadata_path.name],
    )
    print(f"Saved: {report_path}")

    print("\nAnalysis completed successfully.")
    print("Start with these files:")
    print("  02_retention_by_scale_density.csv")
    print("  06_paired_effects_by_scale.csv")
    print("  10_bank_to_bank_distribution_by_scale_density.csv")
    print("  12_density_p50_vs_p05_effects.csv")
    print("  13_article_interbank_table.csv")
    print("  16_reference_density_article_table.csv")
    print("  15_READ_ME_FIRST.md")


if __name__ == "__main__":
    main()