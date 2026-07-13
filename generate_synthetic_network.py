from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

SECTOR_ORDER = [
    "fossil_fuels",
    "utilities",
    "energy_intensive",
    "transportation",
    "buildings_real_estate",
    "agriculture",
    "other_services",
]

STYLIZED_SECTOR_PROBS = {
    "fossil_fuels": 0.08,
    "utilities": 0.10,
    "energy_intensive": 0.18,
    "transportation": 0.14,
    "buildings_real_estate": 0.16,
    "agriculture": 0.14,
    "other_services": 0.20,
}

def make_sector_probabilities(
    distribution: str = "uniform",
    custom_probs: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Return company-sector probabilities for the synthetic economy.

    stylized: original non-uniform sector mix.
    uniform: each sector has the same probability, used for robustness checks.
    """
    if distribution == "stylized":
        probs = STYLIZED_SECTOR_PROBS.copy()

    elif distribution == "uniform":
        probs = {sector: 1.0 / len(SECTOR_ORDER) for sector in SECTOR_ORDER}

    elif distribution == "custom":
        if custom_probs is None:
            raise ValueError("custom_probs must be provided when distribution='custom'.")
        probs = custom_probs.copy()

    else:
        raise ValueError(
            f"Unknown sector distribution: {distribution}. "
            "Use 'stylized', 'uniform', or 'custom'."
        )

    missing = set(SECTOR_ORDER) - set(probs)
    extra = set(probs) - set(SECTOR_ORDER)
    if missing or extra:
        raise ValueError(
            "Sector probability keys must match SECTOR_ORDER. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}"
        )

    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Sector probabilities must sum to a positive number.")

    return {sector: float(probs[sector]) / total for sector in SECTOR_ORDER}


def density_label_from_probability(edge_probability: float) -> str:
    """Convert 0.15 -> p15, 0.30 -> p30 safely."""
    return f"p{int(round(edge_probability * 100)):02d}"


def make_sectors() -> pd.DataFrame:
    return pd.DataFrame({
        "sector": SECTOR_ORDER,
        "dominant_climate_risk_type": [
            "transition",
            "transition",
            "transition",
            "transition",
            "mixed",
            "physical",
            "low",
        ],
        "transition_base_shock": [0.30, 0.28, 0.22, 0.18, 0.15, 0.10, 0.05],
        "physical_base_shock": [0.05, 0.10, 0.06, 0.08, 0.15, 0.20, 0.03],
    })


def make_scenarios() -> pd.DataFrame:
    """
    Scenario definitions used as metadata and for backward compatibility.

    The empirical shock rates used in the main pipeline come from
    country_sector_shocks.csv. The multiplier columns are kept so older
    validation and diagnostic functions still work.
    """
    transition_intensity = [0.5, 1.0, 1.5, 0.0, 1.0]
    physical_intensity = [0.0, 0.0, 0.0, 1.0, 1.0]

    return pd.DataFrame({
        "scenario_id": ["S1", "S2", "S3", "S4", "S5"],
        "scenario_name": [
            "orderly_transition",
            "disorderly_transition",
            "severe_disorderly_transition",
            "acute_physical_risk",
            "combined_transition_physical",
        ],
        "transition_intensity": transition_intensity,
        "physical_intensity": physical_intensity,
        "transition_multiplier": transition_intensity,
        "physical_multiplier": physical_intensity,
    })


transition_sectors = {"fossil_fuels", "utilities", "energy_intensive", "transportation"}
physical_sectors = {"agriculture", "buildings_real_estate"}
mixed_sectors = {"utilities", "transportation", "buildings_real_estate", "agriculture"}


def sector_specialization_boost(bank_type: str, sector: str) -> float:
    """Multiplicative lending preference by bank type and firm sector."""
    if bank_type == "transition_exposed" and sector in transition_sectors:
        return 3.0
    if bank_type == "physical_exposed" and sector in physical_sectors:
        return 3.0
    if bank_type == "mixed_exposed" and sector in mixed_sectors:
        return 2.0
    if bank_type == "diversified":
        return 1.2
    return 0.7


def generate_interbank_edges_for_density(
    banks_base: pd.DataFrame,
    edge_probability: float,
    rng: np.random.Generator,
    interbank_liability_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Generate a directed weighted interbank liability network.

    edge_probability changes the number of counterparties. The total
    interbank liability scale of each debtor is controlled by
    target_interbank_liability_ratio.
    """
    rows: list[dict] = []
    bank_ids = banks_base["bank_id"].to_numpy()
    bank_size = dict(zip(banks_base["bank_id"], banks_base["bank_size_proxy"]))
    target_ratio = dict(zip(banks_base["bank_id"], banks_base["target_interbank_liability_ratio"]))

    for debtor in bank_ids:
        possible_creditors = [b for b in bank_ids if b != debtor]
        selected_creditors = [c for c in possible_creditors if rng.random() < edge_probability]

        if len(selected_creditors) == 0:
            selected_creditors = [rng.choice(possible_creditors)]

        total_interbank_liability = (bank_size[debtor]
                                     * target_ratio[debtor]
                                     * float(interbank_liability_scale)
                                    )
        shares = rng.dirichlet(np.ones(len(selected_creditors)))
        amounts = total_interbank_liability * shares

        for creditor, amount in zip(selected_creditors, amounts):
            rows.append({
                "debtor_id": debtor,
                "creditor_id": creditor,
                "liability_amount": round(float(amount), 2),
            })

    return pd.DataFrame(rows)


def generate_synthetic_network(
    seed: int,
    out_dir: Path,
    n_companies: int = 100,
    n_banks: int = 20,
    edge_probability_values: list[float] | None = None,
    sector_distribution: str = "uniform",
    interbank_liability_scale: float = 1.0,
    custom_sector_probs: dict[str, float] | None = None,
) -> None:
    """
    Generate one independent synthetic climate-financial network.

    One seed = one synthetic financial system. For GNN training, call this
    function many times with different seeds and output folders.
    """
    if edge_probability_values is None:
        edge_probability_values = [0.15, 0.30]

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sectors = make_sectors()
    scenarios = make_scenarios()
    sectors.to_csv(out_dir / "sectors.csv", index=False)
    scenarios.to_csv(out_dir / "scenarios.csv", index=False)

    sector_probs = make_sector_probabilities(
        distribution=sector_distribution,
         custom_probs=custom_sector_probs,
    )

    pd.DataFrame({
        "sector_distribution": sector_distribution,
        "sector": list(sector_probs.keys()),
        "sector_probability": list(sector_probs.values()),
    }).to_csv(out_dir / "sector_distribution.csv", index=False)

    company_sectors = rng.choice(
        list(sector_probs.keys()),
        size=n_companies,
        p=list(sector_probs.values()),
    )

    company_initial_assets = rng.lognormal(mean=np.log(100), sigma=0.55, size=n_companies)
    company_external_assets = company_initial_assets * rng.uniform(0.65, 0.95, size=n_companies)
    company_total_debt = company_external_assets * rng.uniform(0.70, 0.95, size=n_companies)

    companies = pd.DataFrame({
        "company_id": [f"C{i:03d}" for i in range(1, n_companies + 1)],
        "sector": company_sectors,
        "initial_assets": company_initial_assets.round(2),
        "external_assets": company_external_assets.round(2),
        "total_debt": company_total_debt.round(2),
    })
    companies["leverage"] = (companies["total_debt"] / companies["external_assets"]).round(3)
    companies = companies.merge(
        sectors[["sector", "dominant_climate_risk_type", "transition_base_shock", "physical_base_shock"]],
        on="sector",
        how="left",
    )
    if companies["transition_base_shock"].isna().any():
        raise ValueError("Some companies have missing sector shock values.")

    companies.to_csv(out_dir / "companies.csv", index=False)

    if n_banks % 4 != 0:
        raise ValueError("n_banks should be divisible by 4 for equal bank_type groups.")

    bank_size_proxy = rng.lognormal(mean=np.log(900), sigma=0.45, size=n_banks)
    bank_ids = [f"B{i:03d}" for i in range(1, n_banks + 1)]
    group_size = n_banks // 4
    bank_types = (
        ["transition_exposed"] * group_size
        + ["physical_exposed"] * group_size
        + ["mixed_exposed"] * group_size
        + ["diversified"] * group_size
    )

    banks_base = pd.DataFrame({
        "bank_id": bank_ids,
        "bank_size_proxy": bank_size_proxy.round(2),
        "bank_type": bank_types,
    })
    banks_base["external_assets"] = (
        banks_base["bank_size_proxy"] * rng.uniform(0.03, 0.10, size=n_banks)
    ).round(2)
    banks_base["capital_ratio_assumption"] = rng.uniform(0.04, 0.10, size=n_banks).round(3)
    banks_base["capital"] = (
        banks_base["bank_size_proxy"] * banks_base["capital_ratio_assumption"]
    ).round(2)
    banks_base["liquidity_ratio_assumption"] = rng.uniform(0.02, 0.07, size=n_banks).round(3)
    banks_base["liquidity_buffer"] = (
        banks_base["bank_size_proxy"] * banks_base["liquidity_ratio_assumption"]
    ).round(2)
    banks_base["target_interbank_liability_ratio"] = rng.uniform(0.10, 0.30, size=n_banks).round(3)
    banks_base.to_csv(out_dir / "banks_base.csv", index=False)

    firm_bank_edges: list[dict] = []
    for _, firm in companies.iterrows():
        n_lenders = rng.integers(1, 5)
        size_weights = banks_base["bank_size_proxy"].to_numpy(dtype=float)
        specialization_weights = banks_base["bank_type"].apply(
            lambda bank_type: sector_specialization_boost(bank_type, firm["sector"])
        ).to_numpy(dtype=float)

        lender_weights = size_weights * specialization_weights
        lender_weights = lender_weights / lender_weights.sum()

        lenders = rng.choice(
            banks_base["bank_id"].to_numpy(),
            size=n_lenders,
            replace=False,
            p=lender_weights,
        )

        shares = rng.dirichlet(np.ones(n_lenders))
        amounts = firm["total_debt"] * shares

        for lender, amount in zip(lenders, amounts):
            firm_bank_edges.append({
                "debtor_id": firm["company_id"],
                "creditor_id": lender,
                "liability_amount": round(float(amount), 2),
            })

    firm_bank_edges = pd.DataFrame(firm_bank_edges)
    firm_bank_edges.to_csv(out_dir / "firm_bank_edges.csv", index=False)

    firm_debt_check = (
        firm_bank_edges.groupby("debtor_id")["liability_amount"]
        .sum()
        .reindex(companies["company_id"])
        .to_numpy()
    )
    max_firm_debt_gap = np.max(np.abs(firm_debt_check - companies["total_debt"].to_numpy()))
    print(f"[{out_dir.name}] Max rounding gap in firm-bank debt allocation: {max_firm_debt_gap:.4f}")

    summary_rows: list[dict] = []
    for edge_probability in edge_probability_values:
        label = density_label_from_probability(edge_probability)
        network_dir = out_dir / f"interbank_density_{label}"
        network_dir.mkdir(parents=True, exist_ok=True)

        density_seed = seed + int(round(edge_probability * 1000))
        density_rng = np.random.default_rng(density_seed)

        interbank_edges = generate_interbank_edges_for_density(
            banks_base=banks_base,
            edge_probability=edge_probability,
            rng=density_rng,
            interbank_liability_scale=interbank_liability_scale,
        )        
        interbank_edges.to_csv(network_dir / "interbank_edges.csv", index=False)

        banks = banks_base.copy()
        firm_loan_assets = firm_bank_edges.groupby("creditor_id")["liability_amount"].sum()
        interbank_assets = interbank_edges.groupby("creditor_id")["liability_amount"].sum()
        interbank_liabilities = interbank_edges.groupby("debtor_id")["liability_amount"].sum()

        banks["firm_loan_assets"] = banks["bank_id"].map(firm_loan_assets).fillna(0).round(2)
        banks["interbank_assets"] = banks["bank_id"].map(interbank_assets).fillna(0).round(2)
        banks["interbank_liabilities"] = banks["bank_id"].map(interbank_liabilities).fillna(0).round(2)
        banks["initial_assets"] = (
            banks["external_assets"] + banks["firm_loan_assets"] + banks["interbank_assets"]
        ).round(2)
        banks["capital_ratio"] = (banks["capital"] / banks["initial_assets"]).round(3)
        banks.to_csv(network_dir / "banks.csv", index=False)

        possible_directed_edges = n_banks * (n_banks - 1)
        realised_density = len(interbank_edges) / possible_directed_edges
        out_degree = interbank_edges.groupby("debtor_id").size()
        in_degree = interbank_edges.groupby("creditor_id").size()

        summary_rows.append({
            "sector_distribution": sector_distribution,
            "interbank_liability_scale": interbank_liability_scale,
            "network_label": label,
            "edge_probability_input": edge_probability,
            "realised_interbank_edges": len(interbank_edges),
            "realised_density": round(realised_density, 4),
            "mean_out_degree": round(float(out_degree.mean()), 3),
            "mean_in_degree": round(float(in_degree.mean()), 3),
            "total_interbank_liabilities": round(float(interbank_edges["liability_amount"].sum()), 2),
            "output_folder": str(network_dir),
        })

    network_summary = pd.DataFrame(summary_rows)
    network_summary.to_csv(out_dir / "network_summary.csv", index=False)

    portfolio = (
        firm_bank_edges.merge(
            companies[["company_id", "sector"]],
            left_on="debtor_id",
            right_on="company_id",
            how="left",
        )
        .groupby(["creditor_id", "sector"])["liability_amount"]
        .sum()
        .reset_index()
        .rename(columns={"creditor_id": "bank_id", "liability_amount": "sector_exposure"})
    )
    total_by_bank = portfolio.groupby("bank_id")["sector_exposure"].transform("sum")
    portfolio["sector_exposure_share"] = np.where(
        total_by_bank > 0,
        portfolio["sector_exposure"] / total_by_bank,
        0.0,
    )
    portfolio.to_csv(out_dir / "bank_sector_exposures.csv", index=False)

    print(
        f"[{out_dir.name}] Synthetic network generated. "
        f"Companies={len(companies)}, Banks={len(banks_base)}, "
        f"sector_distribution={sector_distribution}"
    )


def main() -> None:
    generate_synthetic_network(
        seed=42,
        out_dir=BASE_DIR / "data" / "synthetic" / "seed_042",
        edge_probability_values=[0.15, 0.30],
        n_companies=100,
        n_banks=20,
        sector_distribution="uniform",
    )


if __name__ == "__main__":
    main()
