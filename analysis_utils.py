from pathlib import Path
import pandas as pd
import numpy as np


def check_required_columns(df: pd.DataFrame, required_columns: list[str], table_name: str) -> None:
    """Check that a dataframe contains all required columns."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing)}")


def check_baseline_stability(
    clearing_baseline: pd.DataFrame,
    bank_baseline: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """Check whether the baseline system is stable before a climate shock."""

    check_required_columns(
        clearing_baseline,
        ["node_type", "payment_shortfall", "relative_payment_shortfall", "payment_default_label"],
        "clearing_baseline",
    )
    check_required_columns(
        bank_baseline,
        ["payment_default_label", "capital_breach_label", "bank_failed_label"],
        "bank_baseline",
    )

    company_defaults = int(
        clearing_baseline.loc[
            clearing_baseline["node_type"] == "company",
            "payment_default_label",
        ].sum()
    )
    bank_payment_defaults = int(bank_baseline["payment_default_label"].sum())
    bank_capital_breaches = int(bank_baseline["capital_breach_label"].sum())
    failed_banks = int(bank_baseline["bank_failed_label"].sum())

    is_stable = (
        company_defaults == 0
        and bank_payment_defaults == 0
        and bank_capital_breaches == 0
        and failed_banks == 0
    )

    summary = {
        "baseline_stable": is_stable,
        "defaulted_companies": company_defaults,
        "payment_default_banks": bank_payment_defaults,
        "capital_breach_banks": bank_capital_breaches,
        "failed_banks": failed_banks,
    }

    if verbose:
        print("\n=== Baseline stability check ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    return summary


def analyze_defaulted_companies(
    clearing_results: pd.DataFrame,
    top_n: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return companies that defaulted after clearing."""

    required = [
        "node_id",
        "node_type",
        "external_assets_before_shock",
        "shock_rate",
        "asset_loss",
        "external_assets_after_shock",
        "total_liabilities",
        "clearing_payment",
        "payment_shortfall",
        "relative_payment_shortfall",
        "payment_default_label",
    ]
    check_required_columns(clearing_results, required, "clearing_results")

    defaulted_companies = clearing_results.loc[
        (clearing_results["node_type"] == "company")
        & (clearing_results["payment_default_label"] == 1)
    ].sort_values(by="payment_shortfall", ascending=False).copy()

    output = defaulted_companies.head(top_n).copy() if top_n is not None else defaulted_companies.copy()

    if verbose:
        print("\n=== Defaulted companies analysis ===")
        print(f"Number of defaulted companies: {len(defaulted_companies)}")
        print(f"Total company payment shortfall: {defaulted_companies['payment_shortfall'].sum():.2f}")
        print(f"Total asset loss among defaulted companies: {defaulted_companies['asset_loss'].sum():.2f}")
        if len(output) > 0:
            print(output[[
                "node_id",
                "shock_rate",
                "asset_loss",
                "external_assets_after_shock",
                "total_liabilities",
                "clearing_payment",
                "payment_shortfall",
                "relative_payment_shortfall",
            ]])
        else:
            print("No company defaults.")

    return output


def analyze_banks_after_shock(
    bank_results: pd.DataFrame,
    top_n: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analyze bank-level losses, vulnerability labels, and failures after a scenario."""

    required = [
        "bank_id",
        "incoming_nominal",
        "incoming_actual",
        "incoming_shortfall",
        "capital",
        "liquidity_buffer",
        "firm_loan_assets",
        "interbank_assets",
        "interbank_liabilities",
        "total_liabilities",
        "clearing_payment",
        "payment_shortfall",
        "payment_default_label",
        "loss_to_capital_ratio",
        "capital_after_loss",
        "vulnerable_label_25",
        "vulnerable_label_50",
        "capital_breach_label",
        "bank_failed_label",
    ]
    check_required_columns(bank_results, required, "bank_results")

    banks_sorted = bank_results.sort_values(by="loss_to_capital_ratio", ascending=False).copy()
    top_banks = banks_sorted.head(top_n).copy()

    if verbose:
        print("\n=== Bank shock analysis ===")
        print(f"Total bank incoming shortfall: {bank_results['incoming_shortfall'].sum():.2f}")
        print(f"Mean loss_to_capital_ratio: {bank_results['loss_to_capital_ratio'].mean():.4f}")
        print(f"Max loss_to_capital_ratio: {bank_results['loss_to_capital_ratio'].max():.4f}")
        print(f"Vulnerable banks >=25% capital loss: {int(bank_results['vulnerable_label_25'].sum())}")
        print(f"Vulnerable banks >=50% capital loss: {int(bank_results['vulnerable_label_50'].sum())}")
        print(f"Capital-breach banks: {int(bank_results['capital_breach_label'].sum())}")
        print(f"Payment-default banks: {int(bank_results['payment_default_label'].sum())}")
        print(f"Failed banks: {int(bank_results['bank_failed_label'].sum())}")
        print(f"\nTop {top_n} banks by loss_to_capital_ratio:")
        print(top_banks[[
            "bank_id",
            "loss_to_capital_ratio",
            "incoming_shortfall",
            "capital",
            "capital_after_loss",
            "vulnerable_label_25",
            "vulnerable_label_50",
            "capital_breach_label",
            "payment_default_label",
            "firm_loan_assets",
            "interbank_assets",
            "interbank_liabilities",
        ]])

    return top_banks


def create_experiment_summary(
    clearing_baseline: pd.DataFrame,
    bank_baseline: pd.DataFrame,
    clearing_shocked: pd.DataFrame,
    bank_shocked: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact summary table for baseline vs shocked scenario."""

    def summarize_state(clearing_df: pd.DataFrame, bank_df: pd.DataFrame, state_name: str) -> dict:
        companies = clearing_df[clearing_df["node_type"] == "company"].copy()
        return {
            "state": state_name,
            "n_company_defaults": int(companies["payment_default_label"].sum()),
            "n_bank_payment_defaults": int(bank_df["payment_default_label"].sum()),
            "n_bank_capital_breaches": int(bank_df["capital_breach_label"].sum()),
            "n_failed_banks": int(bank_df["bank_failed_label"].sum()),
            "n_vulnerable_25_banks": int(bank_df["vulnerable_label_25"].sum()) if "vulnerable_label_25" in bank_df.columns else 0,
            "n_vulnerable_50_banks": int(bank_df["vulnerable_label_50"].sum()) if "vulnerable_label_50" in bank_df.columns else 0,
            "mean_loss_to_capital_ratio": float(bank_df["loss_to_capital_ratio"].mean()) if "loss_to_capital_ratio" in bank_df.columns else 0.0,
            "max_loss_to_capital_ratio": float(bank_df["loss_to_capital_ratio"].max()) if "loss_to_capital_ratio" in bank_df.columns else 0.0,
            "total_company_asset_loss": float(companies["asset_loss"].sum()) if "asset_loss" in companies.columns else 0.0,
            "total_company_payment_shortfall": float(companies["payment_shortfall"].sum()),
            "total_bank_incoming_shortfall": float(bank_df["incoming_shortfall"].sum()),
            "total_bank_payment_shortfall": float(bank_df["payment_shortfall"].sum()),
        }

    return pd.DataFrame([
        summarize_state(clearing_baseline, bank_baseline, "baseline"),
        summarize_state(clearing_shocked, bank_shocked, "shocked"),
    ])


def export_experiment_to_excel(
    clearing_baseline: pd.DataFrame,
    bank_baseline: pd.DataFrame,
    clearing_shocked: pd.DataFrame,
    bank_shocked: pd.DataFrame,
    save_path: str | Path,
) -> None:
    """Save the main experiment outputs to one Excel workbook."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary = create_experiment_summary(clearing_baseline, bank_baseline, clearing_shocked, bank_shocked)

    defaulted_companies = clearing_shocked.loc[
        (clearing_shocked["node_type"] == "company")
        & (clearing_shocked["payment_default_label"] == 1)
    ].sort_values(by="payment_shortfall", ascending=False)

    affected_banks = bank_shocked.sort_values(by="loss_to_capital_ratio", ascending=False)

    with pd.ExcelWriter(save_path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        defaulted_companies.to_excel(writer, sheet_name="defaulted_companies", index=False)
        affected_banks.to_excel(writer, sheet_name="affected_banks", index=False)
        bank_baseline.to_excel(writer, sheet_name="bank_baseline", index=False)
        bank_shocked.to_excel(writer, sheet_name="bank_shocked", index=False)

    print(f"Experiment workbook saved to: {save_path}")

