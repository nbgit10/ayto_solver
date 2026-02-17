#!/usr/bin/env python3
"""Build static JSON data files from AYTO season YAML inputs.

Reads seasons.json, runs GraphSolver on each season, writes JSON to
frontend/public/data/ for the Astro static site.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ayto_solver.solvers.graph_solver import GraphSolver

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "frontend" / "public" / "data"


def load_season_registry() -> list[dict]:
    with open(ROOT / "seasons.json", encoding="utf-8") as f:
        return json.load(f)["seasons"]


def load_input(path: str) -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def solve_season(data: dict) -> dict:
    """Run GraphSolver on season data, return raw results."""
    males = data["MALES"]
    females = data["FEMALES"]
    solver = GraphSolver(males, females)

    for tb in data.get("TRUTH_BOOTH", []):
        solver.add_truth_booth(tb["Pair"][0], tb["Pair"][1], tb["Match"])

    for night in data.get("MATCHING_NIGHTS", []):
        pairs = [(p[0], p[1]) for p in night["Pairs"]]
        solver.add_matching_night(pairs, night["Matches"])

    matchings, capped = solver.enumerate_all_matchings(max_matchings=100000)
    probabilities = solver.calculate_probabilities(matchings)
    double_match_probs = solver.calculate_double_match_probabilities(matchings)

    return {
        "matchings": matchings,
        "capped": capped,
        "probabilities": probabilities,
        "double_match_probs": double_match_probs,
        "solver": solver,
    }


def build_confirmed_and_ruled_out(data: dict) -> tuple[list[dict], list[dict]]:
    """Extract confirmed matches and ruled-out pairs from truth booth data."""
    confirmed = []
    ruled_out = []
    for tb in data.get("TRUTH_BOOTH", []):
        pair = {"male": tb["Pair"][0], "female": tb["Pair"][1]}
        if tb["Match"]:
            confirmed.append(pair)
        else:
            ruled_out.append(pair)
    return confirmed, ruled_out


def build_matching_nights(data: dict) -> list[dict]:
    """Format matching nights for JSON output."""
    nights = []
    for i, night in enumerate(data.get("MATCHING_NIGHTS", []), 1):
        nights.append({
            "night_number": i,
            "pairs": [[p[0], p[1]] for p in night["Pairs"]],
            "matches": night["Matches"],
        })
    return nights


def build_top_matching(
    males: list[str],
    females: list[str],
    matchings: list[set[tuple[str, str]]],
    probabilities: dict,
) -> list[dict]:
    """Find the most probable feasible matching.

    Scores each enumerated solution by the sum of its pair probabilities,
    then returns the highest-scoring one. Guaranteed to be a valid solution.
    """
    if not matchings:
        return []

    # Score each solution: sum of marginal probabilities of its pairs
    best_matching = None
    best_score = -1.0
    for matching in matchings:
        score = sum(probabilities.get(pair, 0.0) for pair in matching)
        if score > best_score:
            best_score = score
            best_matching = matching

    # Detect double-match person (appears in 2+ pairs)
    name_counts: dict[str, int] = defaultdict(int)
    for male, female in best_matching:
        name_counts[male] += 1
        name_counts[female] += 1
    doubled_names = {name for name, count in name_counts.items() if count > 1}

    result = []
    for male, female in best_matching:
        prob = probabilities.get((male, female), 0.0)
        is_double = male in doubled_names or female in doubled_names
        result.append({
            "male": male,
            "female": female,
            "probability": round(prob, 6),
            "is_double": is_double,
        })

    result.sort(key=lambda p: -p["probability"])
    return result


def build_pairings(
    males: list[str],
    females: list[str],
    probabilities: dict,
    confirmed_names: set[tuple[str, str]],
) -> list[dict]:
    """Build full pairings list with probabilities."""
    pairings = []
    for male in males:
        for female in females:
            prob = probabilities.get((male, female), 0.0)
            is_confirmed = (male, female) in confirmed_names
            pairings.append({
                "male": male,
                "female": female,
                "probability": round(prob, 6),
                "confirmed": is_confirmed,
            })
    # Sort: confirmed first, then by probability descending
    pairings.sort(key=lambda p: (-int(p["confirmed"]), -p["probability"]))
    return pairings


def build_double_match(
    solver: GraphSolver,
    double_match_probs: dict,
) -> dict:
    """Build double match info."""
    is_unbalanced = solver.n_males != solver.n_females
    if not is_unbalanced:
        return {"applicable": False, "candidates": []}

    # Determine which gender has double match
    if solver.n_females > solver.n_males:
        double_gender = "male"  # a male gets matched twice
    else:
        double_gender = "female"  # a female gets matched twice

    candidates = []
    for name, prob in sorted(double_match_probs.items(), key=lambda x: -x[1]):
        candidates.append({
            "name": name,
            "probability": round(prob, 6),
            "gender": double_gender,
        })
    return {"applicable": True, "candidates": candidates}


def build_season_json(season: dict, now: str) -> dict | None:
    """Build the full JSON for one season. Returns None if infeasible."""
    data = load_input(season["input_file"])
    males = data["MALES"]
    females = data["FEMALES"]

    result = solve_season(data)
    matchings = result["matchings"]
    solver = result["solver"]

    confirmed, ruled_out = build_confirmed_and_ruled_out(data)
    confirmed_set = {(c["male"], c["female"]) for c in confirmed}

    total_solutions = len(matchings)
    if total_solutions == 0:
        # Infeasible season — still produce a file but mark it
        return {
            "meta": {
                "id": season["id"],
                "display_name": season["display_name"],
                "year": season["year"],
                "type": season["type"],
                "generated_at": now,
                "latest_episode": season["latest_episode"],
                "total_episodes": season["total_episodes"],
            },
            "contestants": {
                "males": males,
                "females": females,
                "is_unbalanced": len(males) != len(females),
            },
            "solver_result": {
                "total_solutions": 0,
                "solutions_capped": False,
                "solved": False,
                "infeasible": True,
            },
            "confirmed_matches": confirmed,
            "ruled_out": ruled_out,
            "pairings": [],
            "double_match": {"applicable": False, "candidates": []},
            "matching_nights": build_matching_nights(data),
        }

    pairings = build_pairings(males, females, result["probabilities"], confirmed_set)
    is_unbalanced = len(males) != len(females)
    top_matching = build_top_matching(
        males, females, matchings, result["probabilities"]
    )

    # A season is "solved" if there's exactly 1 solution
    solved = total_solutions == 1

    return {
        "meta": {
            "id": season["id"],
            "display_name": season["display_name"],
            "year": season["year"],
            "type": season["type"],
            "generated_at": now,
            "latest_episode": season["latest_episode"],
            "total_episodes": season["total_episodes"],
        },
        "contestants": {
            "males": males,
            "females": females,
            "is_unbalanced": len(males) != len(females),
        },
        "solver_result": {
            "total_solutions": total_solutions,
            "solutions_capped": result["capped"],
            "solved": solved,
            "infeasible": False,
        },
        "confirmed_matches": confirmed,
        "ruled_out": ruled_out,
        "pairings": pairings,
        "top_matching": top_matching,
        "double_match": build_double_match(solver, result["double_match_probs"]),
        "matching_nights": build_matching_nights(data),
    }


def main():
    seasons = load_season_registry()
    now = datetime.now(timezone.utc).isoformat()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seasons_index = []

    for season in seasons:
        sid = season["id"]
        print(f"Building {sid}...", end=" ", flush=True)

        season_json = build_season_json(season, now)

        # Write per-season file
        season_dir = OUTPUT_DIR / sid
        season_dir.mkdir(parents=True, exist_ok=True)
        out_path = season_dir / "latest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(season_json, f, ensure_ascii=False, indent=2)

        # Summary for index
        sr = season_json["solver_result"]
        contestants = season_json["contestants"]
        seasons_index.append({
            "id": sid,
            "display_name": season["display_name"],
            "year": season["year"],
            "type": season["type"],
            "current": season.get("current", False),
            "latest_episode": season["latest_episode"],
            "total_episodes": season["total_episodes"],
            "num_males": len(contestants["males"]),
            "num_females": len(contestants["females"]),
            "total_solutions": sr["total_solutions"],
            "confirmed_matches": len(season_json["confirmed_matches"]),
            "infeasible": sr.get("infeasible", False),
            "data_url": f"/data/{sid}/latest.json",
        })

        print(f"{sr['total_solutions']} solutions")

    # Write seasons index
    index = {"generated_at": now, "seasons": seasons_index}
    with open(OUTPUT_DIR / "seasons.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Wrote data for {len(seasons)} seasons to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
