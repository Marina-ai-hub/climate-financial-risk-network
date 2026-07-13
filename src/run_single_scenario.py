from pathlib import Path
import argparse
import pandas as pd

from generate_synthetic_network import generate_synthetic_network
from clearing import (
    validate_input_tables,
    build_nodes,
    build_liability_matrix,
    apply_empirical_climate_scenario,
    eisenberg_noe_clearing,
    create_results_table,
    create_bank_results,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE_DIR / "data" / "synthetic"
OUT_DIR = BASE_DIR / "outputs" / "tables"
SUMMARY_DIR = BASE_DIR / "outputs" / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
CLIMATE_SHOCKS_PATH = (
    BASE_DIR / "data" / "real" / "climate" / "country_sector_shocks.csv"
)


def experiment_data_root(
    sector_distribution: str,
    interbank_liability_scale: float = 1.0,) -> Path:
    ib_label = interbank_scale_label(interbank_liability_scale)

    if sector_distribution == "stylized":
        return DATA_ROOT / ib_label

    return (
        BASE_DIR
        / "data"
        / f"synthetic_{sector_distribution}_sector_distribution"
        / ib_label
    )


def experiment_summary_path(
    sector_distribution: str,
    include_bank_liquidity: bool,
    interbank_liability_scale: float | None = None,
) -> Path:
    liq_label = liquidity_label(include_bank_liquidity)

    if interbank_liability_scale is None:
        ib_part = ""
    else:
        ib_part = f"_{interbank_scale_label(interbank_liability_scale)}"

    if sector_distribution == "stylized":
        return SUMMARY_DIR / f"batch_simulation_summary{ib_part}_{liq_label}.csv"

    return SUMMARY_DIR / (
        f"batch_simulation_summary_"
        f"{sector_distribution}_sector_distribution"
        f"{ib_part}_{liq_label}.csv"
    )


def graph_id_sector_suffix(sector_distribution: str) -> str:
    return "" if sector_distribution == "stylized" else f"_{sector_distribution}_sector_distribution"

def liquidity_label(include_bank_liquidity: bool) -> str:
    return "with_liquidity" if include_bank_liquidity else "no_liquidity"

def interbank_scale_label(scale: float) -> str:
    return "ibx" + str(float(scale)).replace(".", "_")


def compute_shortfall_decomposition(
    nodes: pd.DataFrame,
    L,
    Pi,
    clearing_payments,
) -> dict:
    """
    Decompose incoming payment shortfalls by channel:
    - company -> bank
    - bank -> bank

    L[i, j] is the nominal liability from debtor i to creditor j.
    Actual payment from i to j is clearing_payments[i] * Pi[i, j].
    """

    import numpy as np

    node_types = nodes["node_type"].to_numpy()

    company_idx = np.where(node_types == "company")[0]
    bank_idx = np.where(node_types == "bank")[0]

    actual_payment_matrix = clearing_payments[:, None] * Pi
    shortfall_matrix = np.maximum(L - actual_payment_matrix, 0.0)

    firm_to_bank_nominal = float(L[np.ix_(company_idx, bank_idx)].sum())
    firm_to_bank_actual = float(actual_payment_matrix[np.ix_(company_idx, bank_idx)].sum())
    firm_to_bank_shortfall = float(shortfall_matrix[np.ix_(company_idx, bank_idx)].sum())

    bank_to_bank_nominal = float(L[np.ix_(bank_idx, bank_idx)].sum())
    bank_to_bank_actual = float(actual_payment_matrix[np.ix_(bank_idx, bank_idx)].sum())
    bank_to_bank_shortfall = float(shortfall_matrix[np.ix_(bank_idx, bank_idx)].sum())

    total_bank_incoming_shortfall_decomposed = (
        firm_to_bank_shortfall + bank_to_bank_shortfall
    )

    return {
        "firm_to_bank_nominal_liability": firm_to_bank_nominal,
        "firm_to_bank_actual_payment": firm_to_bank_actual,
        "firm_to_bank_incoming_shortfall": firm_to_bank_shortfall,

        "bank_to_bank_nominal_liability": bank_to_bank_nominal,
        "bank_to_bank_actual_payment": bank_to_bank_actual,
        "bank_to_bank_incoming_shortfall": bank_to_bank_shortfall,

        "total_bank_incoming_shortfall_decomposed": total_bank_incoming_shortfall_decomposed,

        "bank_to_bank_shortfall_share": (
            bank_to_bank_shortfall / total_bank_incoming_shortfall_decomposed
            if total_bank_incoming_shortfall_decomposed > 0
            else 0.0
        ),

        "bank_to_bank_vs_firm_to_bank_shortfall_ratio": (
            bank_to_bank_shortfall / firm_to_bank_shortfall
            if firm_to_bank_shortfall > 0
            else 0.0
        ),
    }

def run_one_experiment(
    seed_id: str,
    data_dir: Path,
    country: str,
    scenario_name: str,
    density_label: str,
    country_sector_shocks: pd.DataFrame,
    include_bank_liquidity: bool = False,
    shock_multiplier: float = 1.0,
    sector_distribution: str = "uniform",
    interbank_liability_scale: float = 1.0,
) -> dict:
    """
    Run one climate scenario on one synthetic network seed and one interbank density.

    Main setting for this project: include_bank_liquidity=False.
    """

    liq_label = liquidity_label(include_bank_liquidity)
    country_safe = country.replace(" ", "_").replace("/", "_")
    shock_label = f"shockx{str(shock_multiplier).replace('.', '_')}"
    ib_label = interbank_scale_label(interbank_liability_scale)

    graph_id = (
        f"{seed_id}_{country_safe}_{scenario_name}_{density_label}_"
        f"{ib_label}_{shock_label}_{liq_label}"
        f"{graph_id_sector_suffix(sector_distribution)}"
    )
    network_dir = data_dir / f"interbank_density_{density_label}"

    sectors = pd.read_csv(data_dir / "sectors.csv")
    scenarios = pd.read_csv(data_dir / "scenarios.csv")
    companies = pd.read_csv(data_dir / "companies.csv")
    firm_bank_edges = pd.read_csv(data_dir / "firm_bank_edges.csv")
    banks = pd.read_csv(network_dir / "banks.csv")
    interbank_edges = pd.read_csv(network_dir / "interbank_edges.csv")

    scenario_rows = scenarios.loc[scenarios["scenario_name"] == scenario_name]
    if scenario_rows.empty:
        available = scenarios["scenario_name"].tolist()
        raise ValueError(f"Unknown scenario_name={scenario_name}. Available scenarios: {available}")
    scenario = scenario_rows.iloc[0]

    validate_input_tables(
        sectors=sectors,
        scenarios=scenarios,
        companies=companies,
        banks=banks,
        firm_bank_edges=firm_bank_edges,
        interbank_edges=interbank_edges,
    )

    nodes = build_nodes(
        companies=companies,
        banks=banks,
        include_bank_liquidity=include_bank_liquidity,
    )
    L, _ = build_liability_matrix(
        nodes=nodes,
        firm_bank_edges=firm_bank_edges,
        interbank_edges=interbank_edges,
    )

    # Baseline clearing before climate shock.
    clearing_baseline, total_liabilities, Pi_baseline, n_iter_baseline = eisenberg_noe_clearing(
        L=L,
        external_assets=nodes["external_assets"].to_numpy(dtype=float),
    )
    baseline_results = create_results_table(nodes, L, clearing_baseline, total_liabilities, Pi_baseline)
    baseline_results["graph_id"] = graph_id
    baseline_results["seed_id"] = seed_id
    baseline_results["scenario_name"] = "baseline_no_climate_shock"
    baseline_results["density_label"] = density_label
    baseline_results["include_bank_liquidity"] = include_bank_liquidity
    baseline_results["liquidity_label"] = liq_label

    baseline_bank_results = create_bank_results(baseline_results, banks)
    baseline_bank_results["graph_id"] = graph_id
    baseline_bank_results["seed_id"] = seed_id
    baseline_bank_results["scenario_name"] = "baseline_no_climate_shock"
    baseline_bank_results["density_label"] = density_label
    baseline_bank_results["include_bank_liquidity"] = include_bank_liquidity
    baseline_bank_results["liquidity_label"] = liq_label

    # Shocked clearing.
    nodes_shocked = apply_empirical_climate_scenario(
        nodes=nodes,
        companies=companies,
        country=country,
        scenario_name=scenario_name,
        country_sector_shocks=country_sector_shocks,
        shock_multiplier=shock_multiplier,
    )
    clearing_shocked, total_liabilities, Pi_shocked, n_iter_shocked = eisenberg_noe_clearing(
        L=L,
        external_assets=nodes_shocked["external_assets"].to_numpy(dtype=float),
    )
    shocked_results = create_results_table(nodes_shocked, 
                                           L, 
                                           clearing_shocked, 
                                           total_liabilities, 
                                           Pi_shocked)
    
    shortfall_decomposition = compute_shortfall_decomposition(
                                            nodes=nodes_shocked,
                                            L=L,
                                            Pi=Pi_shocked,
                                            clearing_payments=clearing_shocked,
                                            )

    shocked_results["graph_id"] = graph_id
    shocked_results["seed_id"] = seed_id
    shocked_results["scenario_name"] = scenario_name
    shocked_results["density_label"] = density_label
    shocked_results["include_bank_liquidity"] = include_bank_liquidity
    shocked_results["liquidity_label"] = liq_label

    shocked_bank_results = create_bank_results(shocked_results, banks)
    shocked_bank_results["graph_id"] = graph_id
    shocked_bank_results["seed_id"] = seed_id
    shocked_bank_results["scenario_name"] = scenario_name
    shocked_bank_results["density_label"] = density_label
    shocked_bank_results["include_bank_liquidity"] = include_bank_liquidity
    shocked_bank_results["liquidity_label"] = liq_label

    # Add scenario metadata before saving, so output CSV files contain country and shock severity.
    for df in [baseline_results, baseline_bank_results, shocked_results, shocked_bank_results]:
        df["country"] = country
        df["shock_multiplier"] = shock_multiplier
        df["shock_label"] = shock_label
        df["interbank_liability_scale"] = interbank_liability_scale
        df["interbank_scale_label"] = ib_label

    # Save outputs. graph_id includes seed, country, scenario, density, shock severity, and liquidity.
    baseline_results.to_csv(OUT_DIR / f"clearing_results_baseline_{graph_id}.csv", index=False)
    baseline_bank_results.to_csv(OUT_DIR / f"bank_results_baseline_{graph_id}.csv", index=False)
    shocked_results.to_csv(OUT_DIR / f"clearing_results_{graph_id}.csv", index=False)
    shocked_bank_results.to_csv(OUT_DIR / f"bank_results_{graph_id}.csv", index=False)

    baseline_company_defaults = int(
        baseline_results.loc[
            baseline_results["node_type"] == "company",
            "payment_default_label",
        ].sum()
    )
    baseline_bank_payment_defaults = int(baseline_bank_results["payment_default_label"].sum())
    baseline_bank_capital_breaches = int(baseline_bank_results["capital_breach_label"].sum())
    baseline_failed_banks = int(baseline_bank_results["bank_failed_label"].sum())

    baseline_stable = (
        baseline_company_defaults == 0
        and baseline_bank_payment_defaults == 0
        and baseline_bank_capital_breaches == 0
        and baseline_failed_banks == 0
    )

    shocked_company_defaults = int(
        shocked_results.loc[
            shocked_results["node_type"] == "company",
            "payment_default_label",
        ].sum()
    )

    baseline_config_id = (
        f"{seed_id}|{density_label}|{ib_label}|"
        f"{sector_distribution}|{liq_label}"
    )

    summary = {
        "baseline_config_id": baseline_config_id,
        "graph_id": graph_id,
        "seed_id": seed_id,
        "country": country,
        "scenario_name": scenario_name,
        "density_label": density_label,
        "sector_distribution": sector_distribution,
        "include_bank_liquidity": include_bank_liquidity,
        "interbank_liability_scale": interbank_liability_scale,
        "interbank_scale_label": ib_label,
        "shock_multiplier": shock_multiplier,
        "shock_label": shock_label,
        "liquidity_label": liq_label,
        "baseline_stable": baseline_stable,
        "baseline_iterations": n_iter_baseline,
        "shocked_iterations": n_iter_shocked,
        "baseline_company_defaults": baseline_company_defaults,
        "baseline_bank_payment_defaults": baseline_bank_payment_defaults,
        "baseline_bank_capital_breaches": baseline_bank_capital_breaches,
        "baseline_failed_banks": baseline_failed_banks,
        "shocked_company_defaults": shocked_company_defaults,
        "shocked_node_defaults": int(shocked_results["payment_default_label"].sum()),
        "shocked_bank_payment_defaults": int(shocked_bank_results["payment_default_label"].sum()),
        "shocked_bank_capital_breaches": int(shocked_bank_results["capital_breach_label"].sum()),
        "shocked_failed_banks": int(shocked_bank_results["bank_failed_label"].sum()),
        "shocked_vulnerable_25_banks": int(shocked_bank_results["vulnerable_label_25"].sum()),
        "shocked_vulnerable_50_banks": int(shocked_bank_results["vulnerable_label_50"].sum()),
        "total_company_asset_loss": float(
            shocked_results.loc[shocked_results["node_type"] == "company", "asset_loss"].sum()
        ),
        "total_company_payment_shortfall": float(
            shocked_results.loc[shocked_results["node_type"] == "company", "payment_shortfall"].sum()
        ),
        "total_bank_incoming_shortfall": float(shocked_bank_results["incoming_shortfall"].sum()),
        "total_bank_payment_shortfall": float(shocked_bank_results["payment_shortfall"].sum()),
        "mean_loss_to_capital_ratio": float(shocked_bank_results["loss_to_capital_ratio"].mean()),
        "median_loss_to_capital_ratio": float(shocked_bank_results["loss_to_capital_ratio"].median()),
        "max_loss_to_capital_ratio": float(shocked_bank_results["loss_to_capital_ratio"].max()),
        **shortfall_decomposition,
    }

    print(
        f"Completed {graph_id} | "
        f"baseline_stable={baseline_stable} | "
        f"company_defaults={summary['shocked_company_defaults']} | "
        f"vuln50={summary['shocked_vulnerable_50_banks']} | "
        f"capital_breaches={summary['shocked_bank_capital_breaches']}"
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sector-distribution",
        choices=["stylized", "uniform"],
        default="uniform",
        help="Company-sector probability assumption for synthetic networks.",
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        #default=2,
        default=20,
        help=(
            "Number of synthetic network seeds to generate. "
            "Default is 2 for a quick test run. Increase to 20 or 50 for final experiments."
        ),
    )

    parser.add_argument(
        "--interbank-liability-scales",
        nargs="+",
        type=float,
        #default=[0.0, 1.0, 2.0] 
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        help=(
            "Grid of interbank liability multipliers for sensitivity analysis. "
            "For a full run, use: 0 0.5 1 1.5 2 3."
        ),
    )

    args = parser.parse_args()

    sector_distribution = args.sector_distribution
    n_seeds = args.n_seeds
    interbank_liability_scales = args.interbank_liability_scales

    country_sector_shocks = pd.read_csv(CLIMATE_SHOCKS_PATH)

    print("Loaded country-sector shocks:")
    print(country_sector_shocks[["country", "sector", "scenario_name", "total_shock"]].head())

    countries = sorted(country_sector_shocks["country"].dropna().unique())

    scenario_names = [
        "orderly_transition",
        "disorderly_transition",
        "severe_disorderly_transition",
        "acute_physical_risk",
        "combined_transition_physical",
    ]

    missing_scenarios = sorted(
        set(scenario_names) - set(country_sector_shocks["scenario_name"].unique())
    )

    if missing_scenarios:
        raise ValueError(
            "country_sector_shocks.csv is missing scenarios: "
            f"{missing_scenarios}"
        )

    print("Countries:", countries)
    print("Sector distribution:", sector_distribution)
    print("Number of seeds:", n_seeds)
    print("Interbank liability scales:", interbank_liability_scales)

    density_values = [0.05, 0.15, 0.30, 0.50]
    density_labels = ["p05", "p15", "p30", "p50"]

    include_bank_liquidity = False

    shock_multipliers = [0.75, 1.0, 1.25, 1.5]

    all_rows: list[dict] = []

    for interbank_liability_scale in interbank_liability_scales:
        ib_label = interbank_scale_label(interbank_liability_scale)

        data_root = experiment_data_root(
            sector_distribution=sector_distribution,
            interbank_liability_scale=interbank_liability_scale,
        )

        scale_rows: list[dict] = []

        print("\n" + "#" * 100)
        print(
            f"STARTING INTERBANK SCALE: {interbank_liability_scale} "
            f"({ib_label})"
        )
        print(f"Data root: {data_root}")
        print("#" * 100)

        for seed in range(n_seeds):
            seed_id = f"seed_{seed:03d}"
            data_dir = data_root / seed_id

            print("\n" + "=" * 90)
            print(
                f"Generating synthetic network for {seed_id} | "
                f"sector_distribution={sector_distribution} | "
                f"interbank_liability_scale={interbank_liability_scale}"
            )
            print("=" * 90)

            generate_synthetic_network(
                seed=seed,
                out_dir=data_dir,
                n_companies=100,
                n_banks=20,
                edge_probability_values=density_values,
                sector_distribution=sector_distribution,
                interbank_liability_scale=interbank_liability_scale,
            )

            for shock_multiplier in shock_multipliers:
                for country in countries:
                    for scenario_name in scenario_names:
                        for density_label in density_labels:
                            row = run_one_experiment(
                                seed_id=seed_id,
                                data_dir=data_dir,
                                country=country,
                                scenario_name=scenario_name,
                                density_label=density_label,
                                country_sector_shocks=country_sector_shocks,
                                shock_multiplier=shock_multiplier,
                                include_bank_liquidity=include_bank_liquidity,
                                sector_distribution=sector_distribution,
                                interbank_liability_scale=interbank_liability_scale,
                            )

                            scale_rows.append(row)
                            all_rows.append(row)

        scale_summary = pd.DataFrame(scale_rows)

        scale_save_path = experiment_summary_path(
            sector_distribution=sector_distribution,
            include_bank_liquidity=include_bank_liquidity,
            interbank_liability_scale=interbank_liability_scale,
        )

        scale_summary.to_csv(scale_save_path, index=False)

        print("\nScale-specific batch simulation summary saved to:")
        print(scale_save_path)
        print("Scale summary shape:", scale_summary.shape)

        if "baseline_stable" in scale_summary.columns:
            print("\nBaseline stability counts for this scale:")
            print(scale_summary["baseline_stable"].value_counts(dropna=False))

        print("\nTarget overview for this scale:")
        print(
            scale_summary[
                [
                    "shocked_bank_capital_breaches",
                    "shocked_vulnerable_25_banks",
                    "shocked_vulnerable_50_banks",
                    "mean_loss_to_capital_ratio",
                    "max_loss_to_capital_ratio",
                ]
            ].describe()
        )

    all_summary = pd.DataFrame(all_rows)

    combined_save_path = SUMMARY_DIR / (
        f"batch_interbank_sensitivity_"
        f"{sector_distribution}_sector_distribution_"
        f"{liquidity_label(include_bank_liquidity)}.csv"
    )

    all_summary.to_csv(combined_save_path, index=False)

    print("\n" + "#" * 100)
    print("COMBINED INTERBANK SENSITIVITY SUMMARY SAVED TO:")
    print(combined_save_path)
    print("Combined summary shape:", all_summary.shape)

    print("\nCombined baseline stability counts:")
    print(all_summary["baseline_stable"].value_counts(dropna=False))

    print("\nCombined target overview:")
    print(
        all_summary[
            [
                "interbank_liability_scale",
                "shocked_bank_capital_breaches",
                "shocked_vulnerable_25_banks",
                "shocked_vulnerable_50_banks",
                "mean_loss_to_capital_ratio",
                "max_loss_to_capital_ratio",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()

