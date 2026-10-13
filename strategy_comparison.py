"""
Strategy Comparison Module
---------------------------
Finds drivers who ran the *same* tyre strategy in a session (same
sequence of compounds, e.g. MEDIUM -> HARD -> HARD) and compares their
degradation rates and race results head-to-head. Holding the strategy
constant is what makes the comparison meaningful - two drivers on
different strategies naturally produce different-looking stint charts
for reasons that have nothing to do with tyre management skill.

Can also be pointed at several sessions at once (see SESSIONS) to spot
strategy patterns that recur across years at the same track.

Requirements: same as tire_degradation_analysis.py
"""

import pandas as pd

from tire_degradation_analysis import load_session, get_driver_stints, fit_degradation

# ---------------- CONFIG ----------------
SESSIONS = [
    (2025, "Bahrain", "R"),
]
MIN_GROUP_SIZE = 2   # only report strategies shared by at least this many drivers
# -----------------------------------------


def get_strategy_signature(laps_df):
    """
    Returns a tuple like ('MEDIUM', 'HARD', 'HARD') describing the order
    of compounds a driver used - the strategy "shape", independent of
    exact stint lengths. Two drivers with the same tuple ran the same
    strategy even if one pitted a lap or two earlier or later than the
    other.
    """
    stint_order = (
        laps_df.sort_values("LapNumber")
        .groupby("Stint")["Compound"]
        .first()
    )
    return tuple(stint_order.tolist())


def get_finishing_positions(session):
    """Maps driver code -> classified finishing position, for context."""
    results = session.results[["Abbreviation", "Position"]].copy()
    return dict(zip(results["Abbreviation"], results["Position"]))


def group_drivers_by_strategy(session, drivers=None):
    """
    Returns {strategy_signature: [(driver, laps_df, degr_df), ...]} for
    every driver in `drivers` (or every driver in the session if
    drivers is None).
    """
    driver_list = drivers if drivers else sorted(session.laps["Driver"].unique())
    groups = {}

    for driver in driver_list:
        laps_df = get_driver_stints(session, driver)
        if laps_df.empty:
            continue
        signature = get_strategy_signature(laps_df)
        degr = fit_degradation(laps_df)
        groups.setdefault(signature, []).append((driver, laps_df, degr))

    return groups


def compare_group(signature, drivers_data, finishing_positions):
    """
    Builds a comparison table for one strategy group: per driver, their
    average degradation slope across all stints and their finishing
    position - so you can see whether better tyre management within an
    identical strategy actually translated into a better result.
    """
    rows = []
    for driver, laps_df, degr in drivers_data:
        if degr.empty:
            continue
        rows.append({
            "Driver": driver,
            "Strategy": " -> ".join(signature),
            "AvgSlopeSecPerLap": round(degr["SlopeSecPerLap"].mean(), 3),
            "TotalLaps": len(laps_df),
            "FinishPosition": finishing_positions.get(driver),
        })
    return pd.DataFrame(rows).sort_values("AvgSlopeSecPerLap")


def analyze_session_strategies(year, gp, session_type, drivers=None):
    session = load_session(year, gp, session_type)
    groups = group_drivers_by_strategy(session, drivers)
    finishing_positions = get_finishing_positions(session)

    reports = []
    for signature, drivers_data in groups.items():
        if len(drivers_data) < MIN_GROUP_SIZE:
            continue
        table = compare_group(signature, drivers_data, finishing_positions)
        if not table.empty:
            table["Year"] = year
            table["GrandPrix"] = gp
            reports.append(table)

    if not reports:
        print(f"No shared strategies (min group size {MIN_GROUP_SIZE}) found for {year} {gp}.")
        return pd.DataFrame()

    return pd.concat(reports, ignore_index=True)


def main():
    all_reports = []
    for year, gp, session_type in SESSIONS:
        print(f"\n=== {year} {gp} ===")
        report = analyze_session_strategies(year, gp, session_type)
        if not report.empty:
            for strategy, group in report.groupby("Strategy"):
                print(f"\nStrategy: {strategy}")
                print(group[["Driver", "AvgSlopeSecPerLap", "TotalLaps", "FinishPosition"]].to_string(index=False))
            all_reports.append(report)

    if all_reports:
        combined = pd.concat(all_reports, ignore_index=True)
        combined.to_csv("strategy_comparison.csv", index=False)
        print("\nSaved: strategy_comparison.csv")


if __name__ == "__main__":
    main()
