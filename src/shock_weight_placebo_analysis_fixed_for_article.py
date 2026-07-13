from pathlib import Path
import argparse
import hashlib

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_VARIANT = "realistic_mechanism"
INTERBANK_SCALE_LABEL = "ibx1_0"

DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "gnn"
    / f"bank_level_tabular_dataset_{DATASET_VARIANT}_{INTERBANK_SCALE_LABEL}.csv"
)


OUT_DIR = BASE_DIR / "outputs" / "metrics" / DATASET_VARIANT

TARGET_COL = "target_loss_to_capital_ratio"

SECTOR_ORDER = [
    "fossil_fuels",
    "utilities",
    "energy_intensive",
    "transportation",
    "buildings_real_estate",
    "agriculture",
    "other_services",
]


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x_series = pd.Series(np.asarray(x, dtype=float))
    y_series = pd.Series(np.asarray(y, dtype=float))

    valid = pd.concat([x_series, y_series], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(valid) < 5:
        return np.nan

    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan

    if method == "spearman":
        left = valid.iloc[:, 0].rank(method="average")
        right = valid.iloc[:, 1].rank(method="average")
        value = left.corr(right, method="pearson")
    else:
        value = valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method)
    return float(value) if pd.notna(value) else np.nan



def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator <= 0.0:
        return np.nan

    numerator = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - numerator / denominator


def split_by_seed(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match the tabular baseline split: train/validation/test by seed_id.

    If seed_id is absent, fall back to a deterministic row split so the
    diagnostic can still run on exported subsets.
    """
    rng = np.random.default_rng(seed)

    if "seed_id" not in df.columns:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_train = int(train_ratio * len(indices))
        n_val = int(val_ratio * len(indices))
        train_idx = indices[:n_train]
        test_idx = indices[n_train + n_val:]
        return train_idx, test_idx

    unique_seeds = np.array(sorted(df["seed_id"].unique()))
    rng.shuffle(unique_seeds)

    n_total = len(unique_seeds)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_seeds = set(unique_seeds[:n_train])
    test_seeds = set(unique_seeds[n_train + n_val:])

    train_idx = df.index[df["seed_id"].isin(train_seeds)].to_numpy()
    test_idx = df.index[df["seed_id"].isin(test_seeds)].to_numpy()

    return train_idx, test_idx


def apply_design_filters(
    df: pd.DataFrame,
    density_label: str | None,
    interbank_liability_scale: float | None,
) -> pd.DataFrame:
    """Restrict diagnostics to a pre-specified experimental calibration."""
    filtered = df

    if density_label is not None:
        require_columns(filtered, ["density_label"], "density filter")
        filtered = filtered.loc[
            filtered["density_label"].astype(str).eq(str(density_label))
        ].copy()

    if interbank_liability_scale is not None:
        require_columns(
            filtered,
            ["interbank_liability_scale"],
            "interbank-liability-scale filter",
        )
        scale = pd.to_numeric(
            filtered["interbank_liability_scale"],
            errors="raise",
        )
        filtered = filtered.loc[
            np.isclose(scale, float(interbank_liability_scale))
        ].copy()

    if filtered.empty:
        raise ValueError(
            "No rows remain after applying the requested design filters."
        )

    return filtered.reset_index(drop=True)


def fit_univariate_ols(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float]:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mask = np.isfinite(x_train) & np.isfinite(y_train)
    x_train = x_train[mask]
    y_train = y_train[mask]

    if len(x_train) == 0:
        return 0.0, 0.0

    x_mean = float(np.mean(x_train))
    y_mean = float(np.mean(y_train))
    x_var = float(np.sum((x_train - x_mean) ** 2))

    if x_var <= 1e-12:
        return y_mean, 0.0

    slope = float(np.sum((x_train - x_mean) * (y_train - y_mean)) / x_var)
    intercept = y_mean - slope * x_mean

    return intercept, slope


def evaluate_feature_vector(
    df: pd.DataFrame,
    feature_name: str,
    feature_values: np.ndarray,
    target_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    y = df[target_col].to_numpy(dtype=float)
    x = np.asarray(feature_values, dtype=float)

    intercept, slope = fit_univariate_ols(x[train_idx], y[train_idx])
    pred_test = intercept + slope * x[test_idx]
    y_test = y[test_idx]

    return {
        "feature": feature_name,
        "n_rows": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "intercept": float(intercept),
        "slope": float(slope),
        "all_pearson": safe_corr(x, y, method="pearson"),
        "all_spearman": safe_corr(x, y, method="spearman"),
        "test_pearson": safe_corr(x[test_idx], y_test, method="pearson"),
        "test_spearman": safe_corr(x[test_idx], y_test, method="spearman"),
        "test_r2": float(r2_score_np(y_test, pred_test)),
        "test_mae": float(np.mean(np.abs(pred_test - y_test))),
        "test_rmse": float(np.sqrt(np.mean((pred_test - y_test) ** 2))),
    }


def infer_capital(df: pd.DataFrame) -> pd.Series | None:
    if "log1p_capital" in df.columns:
        return pd.Series(np.expm1(df["log1p_capital"].astype(float)), index=df.index)

    required = ["total_firm_exposure_from_edges", "firm_exposure_to_capital"]
    if all(col in df.columns for col in required):
        exposure = df["total_firm_exposure_from_edges"].astype(float)
        ratio = df["firm_exposure_to_capital"].astype(float)
        capital = pd.Series(np.nan, index=df.index, dtype=float)
        valid = ratio > 0.0
        capital.loc[valid] = exposure.loc[valid] / ratio.loc[valid]
        return capital.replace([np.inf, -np.inf], np.nan)

    return None


def add_diagnostic_indices(
    df: pd.DataFrame,
    use_capital_normalized: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.Series | None]:
    exposure_cols = [f"exposure_{sector}" for sector in SECTOR_ORDER]
    shock_cols = [f"scenario_shock_{sector}" for sector in SECTOR_ORDER]
    require_columns(df, exposure_cols, "sector exposure")
    require_columns(df, shock_cols, "sector shock")
    require_columns(df, [TARGET_COL], "target")

    df = df.copy()
    if (df[TARGET_COL] < -1e-9).any():
        n_negative = int((df[TARGET_COL] < -1e-9).sum())
        raise ValueError(f"{TARGET_COL} contains {n_negative} negative values.")

    df[TARGET_COL] = df[TARGET_COL].clip(lower=0.0)

    exposure_matrix = df[exposure_cols].to_numpy(dtype=float)
    shock_matrix = df[shock_cols].to_numpy(dtype=float)

    total_exposure = exposure_matrix.sum(axis=1)
    actual_weighted = np.sum(exposure_matrix * shock_matrix, axis=1)
    mean_shock = shock_matrix.mean(axis=1)
    uniform_weighted = total_exposure * mean_shock

    capital = infer_capital(df) if use_capital_normalized else None
    valid_capital = capital is not None and capital.notna().any()

    if use_capital_normalized and valid_capital:
        capital_values = capital.to_numpy(dtype=float)
        denominator = np.where(capital_values > 0.0, capital_values, np.nan)

        aggregate_feature = np.divide(
            total_exposure,
            denominator,
            out=np.zeros_like(total_exposure, dtype=float),
            where=np.isfinite(denominator),
        )
        actual_feature = np.divide(
            actual_weighted,
            denominator,
            out=np.zeros_like(actual_weighted, dtype=float),
            where=np.isfinite(denominator),
        )
        uniform_feature = np.divide(
            uniform_weighted,
            denominator,
            out=np.zeros_like(uniform_weighted, dtype=float),
            where=np.isfinite(denominator),
        )

        suffix = "_to_capital"
    else:
        aggregate_feature = total_exposure
        actual_feature = actual_weighted
        uniform_feature = uniform_weighted
        suffix = ""

    feature_vectors = {
        f"aggregate_firm_exposure{suffix}": aggregate_feature,
        f"actual_shock_weighted_sector_exposure{suffix}": actual_feature,
        f"uniform_shock_weighted_sector_exposure{suffix}": uniform_feature,
    }

    df["diagnostic_total_firm_exposure"] = total_exposure
    df["diagnostic_mean_sector_shock"] = mean_shock
    df["diagnostic_actual_shock_weighted_exposure"] = actual_weighted
    df["diagnostic_uniform_shock_weighted_exposure"] = uniform_weighted

    return df, feature_vectors, capital


def choose_randomization_group_cols(df: pd.DataFrame) -> list[str]:
    if "graph_id" in df.columns:
        return ["graph_id"]

    candidates = [
        "seed_id",
        "country",
        "country_name",
        "scenario",
        "scenario_name",
        "density",
        "interbank_density",
        "density_label",
        "shock_multiplier",
        "interbank_liability_scale",
        "interbank_scale",
        "interbank_scale_label",
        "liquidity_label",
    ]

    group_cols = [col for col in candidates if col in df.columns]

    if not group_cols:
        raise ValueError(
            "No grouping columns found for placebo permutation. "
            "Expected graph_id or scenario-identifying columns."
        )

    return group_cols


def stable_seed(base_seed: int, draw_id: int, group_key: object) -> int:
    payload = f"{base_seed}|{draw_id}|{group_key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % (2**32)


def validate_group_shock_consistency(df: pd.DataFrame, group_cols: list[str]) -> None:
    shock_cols = [f"scenario_shock_{sector}" for sector in SECTOR_ORDER]

    if not group_cols:
        return

    bad_groups = 0

    for _, group in df.groupby(group_cols, sort=False, dropna=False):
        unique_shocks = group[shock_cols].drop_duplicates()
        if len(unique_shocks) > 1:
            bad_groups += 1

    if bad_groups > 0:
        raise ValueError(
            f"{bad_groups} randomization groups contain more than one sector-shock vector. "
            "Use more specific group columns, e.g. graph_id or seed/country/scenario/shock_multiplier/density."
        )


def build_random_permuted_feature(
    df: pd.DataFrame,
    draw_id: int,
    seed: int,
    capital: pd.Series | None,
    use_capital_normalized: bool,
) -> np.ndarray:
    exposure_cols = [f"exposure_{sector}" for sector in SECTOR_ORDER]
    shock_cols = [f"scenario_shock_{sector}" for sector in SECTOR_ORDER]

    exposure_matrix = df[exposure_cols].to_numpy(dtype=float)
    shock_matrix = df[shock_cols].to_numpy(dtype=float)
    randomized = np.zeros(len(df), dtype=float)

    group_cols = choose_randomization_group_cols(df)

    if group_cols:
        grouped = df.groupby(group_cols, sort=False, dropna=False).indices.items()
    else:
        grouped = [(("all",), np.arange(len(df)))]

    for group_key, index_values in grouped:
        index_values = np.asarray(index_values, dtype=int)
        rng = np.random.default_rng(stable_seed(seed, draw_id, group_key))

        shocks = shock_matrix[index_values[0], :]
        permuted_shocks = shocks[rng.permutation(len(shocks))]
        randomized[index_values] = (
            exposure_matrix[index_values, :] * permuted_shocks
        ).sum(axis=1)

    if use_capital_normalized and capital is not None and capital.notna().any():
        capital_values = capital.to_numpy(dtype=float)
        denominator = np.where(capital_values > 0.0, capital_values, np.nan)
        randomized = np.divide(
            randomized,
            denominator,
            out=np.zeros_like(randomized, dtype=float),
            where=np.isfinite(denominator),
        )

    return randomized


def summarize_random_placebo(
    random_results: pd.DataFrame,
    actual_metrics: dict,
) -> pd.DataFrame:
    if len(random_results) == 0:
        return pd.DataFrame()

    metric_cols = [
        "all_pearson",
        "all_spearman",
        "test_pearson",
        "test_spearman",
        "test_r2",
        "test_mae",
        "test_rmse",
    ]

    rows = []
    for metric in metric_cols:
        random_values = random_results[metric].dropna().to_numpy(dtype=float)
        actual_value = float(actual_metrics[metric])

        if len(random_values) == 0:
            continue

        lower_is_better = metric in {"test_mae", "test_rmse"}
        if lower_is_better:
            p_value = (1.0 + np.sum(random_values <= actual_value)) / (
                len(random_values) + 1.0
            )
        else:
            p_value = (1.0 + np.sum(random_values >= actual_value)) / (
                len(random_values) + 1.0
            )

        rows.append(
            {
                "metric": metric,
                "actual_value": actual_value,
                "random_mean": float(np.mean(random_values)),
                "random_std": float(np.std(random_values, ddof=1)),
                "random_p05": float(np.quantile(random_values, 0.05)),
                "random_p50": float(np.quantile(random_values, 0.50)),
                "random_p95": float(np.quantile(random_values, 0.95)),
                "placebo_p_value_as_good_or_better": float(p_value),
            }
        )

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if len(df) == 0:
        return "No rows."

    table = df[columns].copy()
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda value: f"{float(value):.6g}")
        else:
            table[col] = table[col].astype(str)

    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in table.astype(str).itertuples(index=False, name=None)
    ]

    return "\n".join([header, separator] + rows)


def write_markdown_report(
    report_path: Path,
    feature_metrics: pd.DataFrame,
    random_summary: pd.DataFrame,
    n_random_draws: int,
    group_cols: list[str],
) -> None:
    display_cols = [
        "feature",
        "all_spearman",
        "test_spearman",
        "test_r2",
        "test_mae",
        "test_rmse",
    ]

    lines = [
        "# Shock-weight placebo robustness check",
        "",
        "This diagnostic compares the real sector-shock mapping with two benchmarks:",
        "",
        "- aggregate firm exposure only;",
        "- a uniform-shock exposure index, where every sector receives the same scenario-average shock;",
        "- randomized placebo mappings, where real sector shocks are permuted across sector labels.",
        "",
        "The target is not regenerated here. This is therefore a placebo/alignment test on the existing scenarios, not a full counterfactual clearing simulation.",
        "",
        f"Random placebo draws: {n_random_draws}",
        f"Randomization groups: {', '.join(group_cols) if group_cols else 'all rows'}",
        "",
        "## Main feature comparison",
        "",
        markdown_table(feature_metrics, display_cols),
        "",
        "## Random placebo summary",
        "",
        markdown_table(random_summary, list(random_summary.columns)),
        "",
        "## Interpretation guide",
        "",
        "Evidence is stronger if the actual shock-weighted sector exposure beats aggregate exposure, beats the uniform-shock benchmark, and lies above the randomized placebo distribution for correlation/R2 metrics.",
        "",
        "If the uniform benchmark performs almost the same as the actual shock-weighted measure, the result is mostly aggregate exposure plus average scenario severity.",
        "",
        "If randomized mappings perform almost as well as the actual mapping, the conclusion should be weakened because sector labels are not adding much information.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-draws", type=int, default=1000)
    parser.add_argument(
        "--raw-exposure",
        action="store_true",
        help="Use raw exposure instead of exposure-to-capital indices.",
    )
    parser.add_argument(
        "--density-label",
        type=str,
        default=None,
        help="Optional density_label filter, e.g. p30 for the article reference calibration.",
    )
    parser.add_argument(
        "--interbank-liability-scale",
        type=float,
        default=None,
        help="Optional interbank_liability_scale filter, e.g. 1.0.",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Tabular dataset not found: {args.data_path}. "
            "Run build_tabular_dataset_fixed_for_article.py first or pass --data-path."
        )

    df = pd.read_csv(args.data_path).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = apply_design_filters(
        df,
        density_label=args.density_label,
        interbank_liability_scale=args.interbank_liability_scale,
    )
    print(f"Rows after design filtering: {len(df):,}")


    required_cols = (
        [TARGET_COL]
        + [f"exposure_{sector}" for sector in SECTOR_ORDER]
        + [f"scenario_shock_{sector}" for sector in SECTOR_ORDER]
    )

    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    nan_required = [col for col in required_cols if df[col].isna().any()]
    if nan_required:
       raise ValueError(f"Required columns contain NaN values: {nan_required}")


    use_capital_normalized = not args.raw_exposure
    df, feature_vectors, capital = add_diagnostic_indices(
        df,
        use_capital_normalized=use_capital_normalized,
    )

    train_idx, test_idx = split_by_seed(df, seed=args.seed)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Train/test split is empty. Check that the dataset has enough "
            "distinct seed_id values, or run on a larger exported table."
        )

    feature_rows = []
    for feature_name, values in feature_vectors.items():
        feature_rows.append(
            evaluate_feature_vector(
                df,
                feature_name,
                values,
                TARGET_COL,
                train_idx,
                test_idx,
            )
        )

    feature_metrics = pd.DataFrame(feature_rows)

    actual_feature_name = [
        name for name in feature_vectors
        if name.startswith("actual_shock_weighted_sector_exposure")
    ][0]
    actual_metrics = feature_metrics[
        feature_metrics["feature"] == actual_feature_name
    ].iloc[0].to_dict()

    random_rows = []
    for draw_id in range(args.random_draws):
        random_values = build_random_permuted_feature(
            df,
            draw_id=draw_id,
            seed=args.seed,
            capital=capital,
            use_capital_normalized=use_capital_normalized,
        )
        row = evaluate_feature_vector(
            df,
            "random_permuted_shock_weighted_sector_exposure",
            random_values,
            TARGET_COL,
            train_idx,
            test_idx,
        )
        row["draw_id"] = draw_id
        random_rows.append(row)

    random_results = pd.DataFrame(random_rows)
    random_summary = summarize_random_placebo(random_results, actual_metrics)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    exposure_label = "raw" if args.raw_exposure else "capital_normalized"
    output_parts = [exposure_label]
    if args.density_label is not None:
        output_parts.append(f"density_{args.density_label}")
    if args.interbank_liability_scale is not None:
        scale_label = str(args.interbank_liability_scale).replace(".", "_")
        output_parts.append(f"ibx{scale_label}")
    output_label = "_".join(output_parts)

    feature_metrics_path = (
        args.out_dir / f"shock_weight_placebo_feature_metrics_{output_label}.csv"
    )
    random_draws_path = (
        args.out_dir / f"shock_weight_placebo_random_draws_{output_label}.csv"
    )
    random_summary_path = (
        args.out_dir / f"shock_weight_placebo_random_summary_{output_label}.csv"
    )
    report_path = (
        args.out_dir / f"shock_weight_placebo_report_{output_label}.md"
    )

    feature_metrics.to_csv(feature_metrics_path, index=False)
    random_results.to_csv(random_draws_path, index=False)
    random_summary.to_csv(random_summary_path, index=False)

    group_cols = choose_randomization_group_cols(df)
    validate_group_shock_consistency(df, group_cols)
    write_markdown_report(
        report_path,
        feature_metrics,
        random_summary,
        n_random_draws=args.random_draws,
        group_cols=group_cols,
    )

    print(f"Saved feature comparison to: {feature_metrics_path}")
    print(f"Saved random placebo draws to: {random_draws_path}")
    print(f"Saved random placebo summary to: {random_summary_path}")
    print(f"Saved markdown report to: {report_path}")
    print()
    print(feature_metrics)
    print()
    print(random_summary)


if __name__ == "__main__":
    main()
