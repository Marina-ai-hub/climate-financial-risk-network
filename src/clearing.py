import numpy as np
import pandas as pd


def validate_input_tables(
    sectors: pd.DataFrame,
    scenarios: pd.DataFrame,
    companies: pd.DataFrame,
    banks: pd.DataFrame,
    firm_bank_edges: pd.DataFrame,
    interbank_edges: pd.DataFrame,
) -> None:
    """
    Validate the input tables before running Eisenberg-Noe clearing.

    Expected edge directions:
    - firm_bank_edges: company debtor -> bank creditor
    - interbank_edges: bank debtor -> bank creditor
    """

    required_sector_cols = {
        "sector",
        "dominant_climate_risk_type",
        "transition_base_shock",
        "physical_base_shock",
    }

    required_scenario_cols = {
        "scenario_id",
        "scenario_name",
        "transition_multiplier",
        "physical_multiplier",
    }

    required_company_cols = {
        "company_id",
        "sector",
        "external_assets",
        "total_debt",
        "transition_base_shock",
        "physical_base_shock",
    }

    required_bank_cols = {
        "bank_id",
        "external_assets",
        "capital",
        "liquidity_buffer",
        "firm_loan_assets",
        "interbank_assets",
        "interbank_liabilities",
        "capital_ratio",
    }

    required_edge_cols = {
        "debtor_id",
        "creditor_id",
        "liability_amount",
    }

    checks = [
        ("sectors", sectors, required_sector_cols),
        ("scenarios", scenarios, required_scenario_cols),
        ("companies", companies, required_company_cols),
        ("banks", banks, required_bank_cols),
        ("firm_bank_edges", firm_bank_edges, required_edge_cols),
        ("interbank_edges", interbank_edges, required_edge_cols),
    ]

    for name, df, required_cols in checks:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    for edge_name, edges in [
        ("firm_bank_edges", firm_bank_edges),
        ("interbank_edges", interbank_edges),
    ]:
        if edges.empty:
            raise ValueError(f"{edge_name} is empty.")
        if edges[["debtor_id", "creditor_id", "liability_amount"]].isna().any().any():
            raise ValueError(f"{edge_name} contains missing debtor, creditor, or amount values.")
        if (edges["liability_amount"] < 0).any():
            raise ValueError(f"{edge_name} contains negative liability_amount values.")

    company_ids = set(companies["company_id"])
    bank_ids = set(banks["bank_id"])
    known_nodes = company_ids | bank_ids

    for edge_name, edges in [
        ("firm_bank_edges", firm_bank_edges),
        ("interbank_edges", interbank_edges),
    ]:
        unknown_debtors = set(edges["debtor_id"]) - known_nodes
        unknown_creditors = set(edges["creditor_id"]) - known_nodes
        if unknown_debtors:
            raise ValueError(f"{edge_name} has unknown debtors: {sorted(unknown_debtors)}")
        if unknown_creditors:
            raise ValueError(f"{edge_name} has unknown creditors: {sorted(unknown_creditors)}")

    # Structural validation of the two edge layers.
    if not set(firm_bank_edges["debtor_id"]).issubset(company_ids):
        raise ValueError("firm_bank_edges must have only companies as debtors.")
    if not set(firm_bank_edges["creditor_id"]).issubset(bank_ids):
        raise ValueError("firm_bank_edges must have only banks as creditors.")
    if not set(interbank_edges["debtor_id"]).issubset(bank_ids):
        raise ValueError("interbank_edges must have only banks as debtors.")
    if not set(interbank_edges["creditor_id"]).issubset(bank_ids):
        raise ValueError("interbank_edges must have only banks as creditors.")
    if (interbank_edges["debtor_id"] == interbank_edges["creditor_id"]).any():
        raise ValueError("interbank_edges contains self-loops.")

    if companies["sector"].isna().any():
        raise ValueError("companies contains missing sector values.")
    if not set(companies["sector"]).issubset(set(sectors["sector"])):
        unknown = set(companies["sector"]) - set(sectors["sector"])
        raise ValueError(f"companies contains sectors not present in sectors table: {sorted(unknown)}")


def build_nodes(
    companies: pd.DataFrame,
    banks: pd.DataFrame,
    include_bank_liquidity: bool = False,
) -> pd.DataFrame:
    """
    Combine companies and banks into one node table.

    For companies, external_assets are the resources available for repayment.
    For banks, external_assets are non-network payment resources. If
    include_bank_liquidity=True, liquidity_buffer is added as immediately
    available payment liquidity. For the main stress-test setting, use False.
    """

    company_nodes = pd.DataFrame({
        "node_id": companies["company_id"],
        "node_type": "company",
        "external_assets": companies["external_assets"].astype(float),
    })

    bank_payment_assets = banks["external_assets"].astype(float).copy()
    if include_bank_liquidity:
        bank_payment_assets = bank_payment_assets + banks["liquidity_buffer"].astype(float)

    bank_nodes = pd.DataFrame({
        "node_id": banks["bank_id"],
        "node_type": "bank",
        "external_assets": bank_payment_assets,
    })

    nodes = pd.concat([company_nodes, bank_nodes], ignore_index=True)
    nodes["external_assets_before_shock"] = nodes["external_assets"].astype(float)
    nodes["shock_rate"] = 0.0
    nodes["asset_loss"] = 0.0
    nodes["external_assets_after_shock"] = nodes["external_assets"].astype(float)

    return nodes


def build_liability_matrix(
    nodes: pd.DataFrame,
    firm_bank_edges: pd.DataFrame,
    interbank_edges: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build liability matrix L, where L[i, j] is the amount node i owes to node j.
    """

    node_ids = nodes["node_id"].tolist()
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    L = np.zeros((len(node_ids), len(node_ids)), dtype=float)

    all_edges = pd.concat([firm_bank_edges, interbank_edges], ignore_index=True)
    for _, row in all_edges.iterrows():
        debtor = row["debtor_id"]
        creditor = row["creditor_id"]
        amount = float(row["liability_amount"])
        if debtor not in node_to_idx:
            raise ValueError(f"Unknown debtor_id: {debtor}")
        if creditor not in node_to_idx:
            raise ValueError(f"Unknown creditor_id: {creditor}")
        L[node_to_idx[debtor], node_to_idx[creditor]] += amount

    return L, node_to_idx


def apply_climate_scenario(
    nodes: pd.DataFrame,
    companies: pd.DataFrame,
    scenario: pd.Series,
    max_shock_rate: float = 0.95,
) -> pd.DataFrame:
    """
    Apply one climate scenario to company external assets.

    Banks are not directly shocked in this prototype. They are affected
    indirectly when companies cannot fully repay them.
    """

    nodes_shocked = nodes.copy()
    transition_multiplier = float(scenario["transition_multiplier"])
    physical_multiplier = float(scenario["physical_multiplier"])

    company_info = companies.copy()
    company_info["external_assets_before_shock"] = company_info["external_assets"].astype(float)
    company_info["shock_rate"] = (
        company_info["transition_base_shock"].astype(float) * transition_multiplier
        + company_info["physical_base_shock"].astype(float) * physical_multiplier
    ).clip(lower=0.0, upper=max_shock_rate)
    company_info["asset_loss"] = company_info["external_assets_before_shock"] * company_info["shock_rate"]
    company_info["external_assets_after_shock"] = (
        company_info["external_assets_before_shock"] - company_info["asset_loss"]
    )

    shock_info = company_info.set_index("company_id")[[
        "external_assets_before_shock",
        "shock_rate",
        "asset_loss",
        "external_assets_after_shock",
    ]]

    is_company = nodes_shocked["node_id"].isin(shock_info.index)
    for col in shock_info.columns:
        nodes_shocked.loc[is_company, col] = (
            nodes_shocked.loc[is_company, "node_id"].map(shock_info[col]).astype(float)
        )

    # This column is the payment resource vector used by Eisenberg-Noe.
    nodes_shocked.loc[is_company, "external_assets"] = nodes_shocked.loc[
        is_company, "external_assets_after_shock"
    ]

    return nodes_shocked

def apply_empirical_climate_scenario(
    nodes: pd.DataFrame,
    companies: pd.DataFrame,
    country: str,
    scenario_name: str,
    country_sector_shocks: pd.DataFrame,
    max_shock_rate: float = 0.95,
    shock_multiplier: float = 1.0
) -> pd.DataFrame:
    """
    Apply empirical country-sector climate shock to company external assets.

    Banks are not directly shocked.
    Banks are affected indirectly through company payment shortfalls.
    """

    nodes_shocked = nodes.copy()

    shocks_for_case = country_sector_shocks.loc[
        (country_sector_shocks["country"] == country)
        & (country_sector_shocks["scenario_name"] == scenario_name)
    ].copy()

    if shocks_for_case.empty:
        available = country_sector_shocks[["country", "scenario_name"]].drop_duplicates()
        raise ValueError(
            f"No shocks found for country={country}, scenario_name={scenario_name}. "
            f"Available combinations:\n{available.head(20)}"
        )

    shock_lookup = shocks_for_case.set_index("sector")["total_shock"].to_dict()

    company_info = companies.copy()

    company_info["external_assets_before_shock"] = (
        company_info["external_assets"].astype(float)
    )

    company_info["shock_rate"] = (
        company_info["sector"]
        .map(shock_lookup)
        .astype(float)
        .mul(shock_multiplier)
        .clip(lower=0.0, upper=max_shock_rate)
    )

    if company_info["shock_rate"].isna().any():
        missing_sectors = sorted(
            company_info.loc[company_info["shock_rate"].isna(), "sector"].unique()
        )
        raise ValueError(
            f"Missing empirical shock for country={country}, "
            f"scenario_name={scenario_name}, sectors={missing_sectors}"
        )

    company_info["asset_loss"] = (
        company_info["external_assets_before_shock"]
        * company_info["shock_rate"]
    )

    company_info["external_assets_after_shock"] = (
        company_info["external_assets_before_shock"]
        - company_info["asset_loss"]
    ).clip(lower=0.0)

    shock_info = company_info.set_index("company_id")[[
        "external_assets_before_shock",
        "shock_rate",
        "asset_loss",
        "external_assets_after_shock",
    ]]

    is_company = nodes_shocked["node_id"].isin(shock_info.index)

    for col in shock_info.columns:
        nodes_shocked.loc[is_company, col] = (
            nodes_shocked.loc[is_company, "node_id"]
            .map(shock_info[col])
            .astype(float)
        )

    nodes_shocked.loc[is_company, "external_assets"] = (
        nodes_shocked.loc[is_company, "external_assets_after_shock"]
    )

    return nodes_shocked


def eisenberg_noe_clearing(
    L: np.ndarray,
    external_assets: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute Eisenberg-Noe clearing payments using fixed-point iteration.
    """

    total_liabilities = L.sum(axis=1)
    Pi = np.zeros_like(L)
    nonzero = total_liabilities > 0
    Pi[nonzero] = L[nonzero] / total_liabilities[nonzero, None]

    p = total_liabilities.copy()
    for iteration in range(max_iter):
        incoming_payments = Pi.T @ p
        p_new = np.minimum(total_liabilities, external_assets + incoming_payments)
        diff = np.max(np.abs(p_new - p)) if len(p_new) else 0.0
        if diff < tol:
            return p_new, total_liabilities, Pi, iteration + 1
        p = p_new

    raise RuntimeError("Eisenberg-Noe clearing did not converge.")


def create_results_table(
    nodes: pd.DataFrame,
    L: np.ndarray,
    clearing_payments: np.ndarray,
    total_liabilities: np.ndarray,
    Pi: np.ndarray,
) -> pd.DataFrame:
    """Create a readable clearing result table for all nodes."""

    incoming_actual = Pi.T @ clearing_payments
    incoming_nominal = L.sum(axis=0)
    incoming_shortfall = np.maximum(incoming_nominal - incoming_actual, 0.0)

    results = nodes.copy()
    for col, default in [
        ("external_assets_before_shock", results["external_assets"].astype(float)),
        ("shock_rate", 0.0),
        ("asset_loss", 0.0),
        ("external_assets_after_shock", results["external_assets"].astype(float)),
    ]:
        if col not in results.columns:
            results[col] = default

    results["incoming_nominal"] = incoming_nominal
    results["incoming_actual"] = incoming_actual
    results["incoming_shortfall"] = incoming_shortfall
    results["total_liabilities"] = total_liabilities
    results["clearing_payment"] = clearing_payments
    results["payment_shortfall"] = np.maximum(total_liabilities - clearing_payments, 0.0)
    results["relative_payment_shortfall"] = np.where(
        total_liabilities > 0,
        results["payment_shortfall"] / total_liabilities,
        0.0,
    )
    results["equity_after_clearing"] = (
        results["external_assets_after_shock"]
        + results["incoming_actual"]
        - results["clearing_payment"]
    )
    results["payment_default_label"] = (
        results["clearing_payment"] < results["total_liabilities"] - 1e-8
    ).astype(int)

    return results


def create_bank_results(results: pd.DataFrame, banks: pd.DataFrame) -> pd.DataFrame:
    """
    Extract bank-only results and create bank-level targets.

    Main GNN target:
    - loss_to_capital_ratio = incoming_shortfall / capital

    Auxiliary labels:
    - vulnerable_label_25, vulnerable_label_50
    - capital_breach_label
    - payment_default_label
    """

    bank_results = results[results["node_type"] == "bank"].copy()
    bank_results = bank_results.rename(columns={"node_id": "bank_id"})

    bank_cols = [
        "bank_id",
        "capital",
        "liquidity_buffer",
        "firm_loan_assets",
        "interbank_assets",
        "interbank_liabilities",
        "capital_ratio",
    ]
    optional_cols = ["bank_type", "bank_size_proxy", "external_assets"]
    bank_cols += [c for c in optional_cols if c in banks.columns]

    bank_results = bank_results.merge(
        banks[bank_cols],
        on="bank_id",
        how="left",
        suffixes=("", "_bank_table"),
    )

    if bank_results["capital"].isna().any():
        raise ValueError("Some bank results have missing capital after merge.")

    bank_results["loss_to_capital_ratio"] = np.where(
        bank_results["capital"] > 0,
        bank_results["incoming_shortfall"] / bank_results["capital"],
        0.0,
    )
    bank_results["capital_after_loss"] = np.maximum(bank_results["capital"] - bank_results["incoming_shortfall"], 0.0)
    bank_results["capital_breach_label"] = (
        bank_results["incoming_shortfall"] > bank_results["capital"]
    ).astype(int)
    bank_results["bank_failed_label"] = (
        (bank_results["payment_default_label"] == 1)
        | (bank_results["capital_breach_label"] == 1)
    ).astype(int)
    bank_results["vulnerable_label_25"] = (
        bank_results["loss_to_capital_ratio"] >= 0.25
    ).astype(int)
    bank_results["vulnerable_label_50"] = (
        bank_results["loss_to_capital_ratio"] >= 0.50
    ).astype(int)

    # Analysis-only distance to payment default. Not used as a GNN input, because it uses clearing outputs.
    bank_results["payment_capacity_after_clearing"] = (
        bank_results["external_assets_after_shock"] + bank_results["incoming_actual"]
    )
    bank_results["payment_margin_after_clearing"] = (
        bank_results["payment_capacity_after_clearing"] - bank_results["total_liabilities"]
    )
    bank_results["relative_payment_margin_after_clearing"] = np.where(
        bank_results["total_liabilities"] > 0,
        bank_results["payment_margin_after_clearing"] / bank_results["total_liabilities"],
        np.inf,
    )

    return bank_results

