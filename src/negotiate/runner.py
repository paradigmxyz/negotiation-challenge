"""Game runner: orchestrates N negotiation games for a given prompt."""

import asyncio
import hashlib
import random
import statistics
import time
from typing import Optional, Callable

from google import genai

from .engine import (
    END_PROBABILITY,
    GUARANTEED_ROUNDS,
    HARD_MAX_ROUNDS,
    NO_DEAL_PENALTY,
    build_system_prompt,
    build_turn_prompt,
    generate_scenario,
    max_possible,
    score_split,
    validate_offer,
)
from .inference import call_gemini


async def run_game(
    client: genai.Client,
    scenario: dict,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    user_goes_first: bool = True,
) -> dict:
    """Run one full negotiation game.

    Player A always moves first. When user_goes_first=True the user is A;
    when False the user is B and the baseline moves first.
    """
    pool = scenario["pool"]
    vals_user = scenario["valuations_a"]
    vals_baseline = scenario["valuations_b"]

    if user_goes_first:
        vals_a, vals_b = vals_user, vals_baseline
        sys_a = build_system_prompt("A", pool, vals_a, custom_strategy=user_prompt)
        sys_b = build_system_prompt("B", pool, vals_b)
        user_role = "A"
    else:
        vals_a, vals_b = vals_baseline, vals_user
        sys_a = build_system_prompt("A", pool, vals_a)
        sys_b = build_system_prompt("B", pool, vals_b, custom_strategy=user_prompt)
        user_role = "B"

    # Pre-determine the stochastic deadline from the seed.
    # Rounds 1-4 are guaranteed; from round 5 onward, 30% chance to end each round.
    deadline_rng = random.Random(scenario["seed"] * 31 + 777)
    effective_max_rounds = HARD_MAX_ROUNDS
    for r in range(GUARANTEED_ROUNDS + 1, HARD_MAX_ROUNDS + 1):
        if deadline_rng.random() < END_PROBABILITY:
            effective_max_rounds = r
            break

    history: list[dict] = []
    deal_reached = False
    final_round = effective_max_rounds
    user_score = NO_DEAL_PENALTY
    baseline_score = NO_DEAL_PENALTY
    last_proposal: Optional[dict] = None
    last_proposer: Optional[str] = None

    for round_num in range(1, effective_max_rounds + 1):
        # -- Player A turn --
        turn_prompt_a = build_turn_prompt(history, "A", round_num, pool)
        result_a = await call_gemini(client, sys_a, turn_prompt_a, semaphore)

        if result_a is None:
            result_a = {
                "action": "reject",
                "message": "(no response)",
                "reasoning": "",
                "offer": None,
            }

        action_a = result_a["action"]
        offer_a = result_a.get("offer")

        if action_a == "accept":
            if last_proposer != "B" or last_proposal is None:
                action_a = "reject"
                result_a["action"] = "reject"

        if action_a == "propose":
            if offer_a is None or not validate_offer(offer_a, pool):
                action_a = "reject"
                result_a["action"] = "reject"
                offer_a = None
                result_a["offer"] = None

        turn_a = {
            "round": round_num,
            "player": "A",
            "action": action_a,
            "message": result_a["message"],
            "reasoning": result_a["reasoning"],
            "offer": offer_a,
        }
        history.append(turn_a)

        if action_a == "propose" and offer_a:
            last_proposal = offer_a
            last_proposer = "A"

        if action_a == "accept" and last_proposal is not None and last_proposer == "B":
            a_split = last_proposal["their_share"]
            b_split = last_proposal["my_share"]
            deal_reached = True
            final_round = round_num
            break

        # -- Player B turn --
        turn_prompt_b = build_turn_prompt(history, "B", round_num, pool)
        result_b = await call_gemini(client, sys_b, turn_prompt_b, semaphore)

        if result_b is None:
            result_b = {
                "action": "reject",
                "message": "(no response)",
                "reasoning": "",
                "offer": None,
            }

        action_b = result_b["action"]
        offer_b = result_b.get("offer")

        if action_b == "accept":
            if last_proposer != "A" or last_proposal is None:
                action_b = "reject"
                result_b["action"] = "reject"

        if action_b == "propose":
            if offer_b is None or not validate_offer(offer_b, pool):
                action_b = "reject"
                result_b["action"] = "reject"
                offer_b = None
                result_b["offer"] = None

        turn_b = {
            "round": round_num,
            "player": "B",
            "action": action_b,
            "message": result_b["message"],
            "reasoning": result_b["reasoning"],
            "offer": offer_b,
        }
        history.append(turn_b)

        if action_b == "propose" and offer_b:
            last_proposal = offer_b
            last_proposer = "B"

        if action_b == "accept" and last_proposal is not None and last_proposer == "A":
            a_split = last_proposal["my_share"]
            b_split = last_proposal["their_share"]
            deal_reached = True
            final_round = round_num
            break

    # Map A/B splits back to user/baseline based on role
    if deal_reached:
        user_split = a_split if user_role == "A" else b_split
        baseline_split = b_split if user_role == "A" else a_split
        max_user = max_possible(vals_user, pool)
        max_baseline = max_possible(vals_baseline, pool)
        user_score = score_split(vals_user, user_split) / max_user if max_user > 0 else 0.0
        baseline_score = score_split(vals_baseline, baseline_split) / max_baseline if max_baseline > 0 else 0.0

    return {
        "scenario_seed": scenario["seed"],
        "deal_reached": deal_reached,
        "final_round": final_round,
        "user_score": round(user_score, 4),
        "baseline_score": round(baseline_score, 4),
        "turns": history,
        "pool": pool,
        "valuations_user": vals_user,
        "valuations_baseline": vals_baseline,
        "user_role": user_role,
    }


async def run_evaluation(
    prompt: str,
    num_games: int = 10,
    seed: int | None = None,
    concurrency: int = 20,
    on_game_complete: Callable[[int, int, dict], None] | None = None,
) -> dict:
    """Run a full evaluation: N games, collect stats.

    Args:
        prompt: The strategy prompt text.
        num_games: Number of games to play.
        seed: Base seed for reproducibility. None = random.
        concurrency: Max concurrent Gemini API calls.
        on_game_complete: Callback(completed, total, game_result) for progress.

    Returns dict with keys: prompt, games, stats, elapsed.
    """
    client = genai.Client()
    semaphore = asyncio.Semaphore(concurrency)

    if seed is None:
        import random
        seed = random.randint(0, 2**31 - 1)

    def _seed_for(idx: int) -> int:
        return int(hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest(), 16) % (2**31)

    scenarios = [generate_scenario(_seed_for(i)) for i in range(num_games)]

    # Alternate who goes first: even games = user is A, odd = user is B
    tasks = [
        asyncio.ensure_future(
            run_game(client, scenario, prompt, semaphore,
                     user_goes_first=(i % 2 == 0))
        )
        for i, scenario in enumerate(scenarios)
    ]

    results = []
    completed = 0
    t_start = time.monotonic()

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        results.append(result)
        if on_game_complete:
            on_game_complete(completed, num_games, result)

    elapsed = time.monotonic() - t_start

    user_scores = [r["user_score"] for r in results]
    deals = sum(1 for r in results if r["deal_reached"])

    stats = {
        "mean": round(statistics.mean(user_scores), 4),
        "median": round(statistics.median(user_scores), 4),
        "min": round(min(user_scores), 4),
        "max": round(max(user_scores), 4),
        "std": round(statistics.stdev(user_scores), 4) if len(user_scores) > 1 else 0.0,
        "deals_reached": deals,
        "games_played": num_games,
        "deal_rate": round(deals / num_games, 2),
    }

    return {
        "prompt": prompt,
        "seed": seed,
        "games": results,
        "stats": stats,
        "elapsed": round(elapsed, 2),
    }
