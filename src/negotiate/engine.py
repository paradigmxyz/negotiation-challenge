"""Pure game logic: scenario generation, validation, scoring, prompt building.

Extracted from the Optimization Arena negotiation router — no network,
no database, no framework dependencies.
"""

import random

GUARANTEED_ROUNDS = 4
END_PROBABILITY = 0.3
HARD_MAX_ROUNDS = 15
NO_DEAL_PENALTY = -0.5
RESOURCE_TYPES = ["books", "hats", "balls"]
TARGET_VALUE = 100
MAX_VALUATION = 12

RULES_PREAMBLE = """\
You are Player {role} in a resource negotiation game against one other player.

OBJECTIVE:
Divide a shared pool of resources to maximize your score. You have private valuations for each resource — the other player has different valuations that you cannot see.

RESOURCE POOL: {pool_str}
YOUR PRIVATE VALUATIONS: {val_str}

TURN STRUCTURE:
- Each round, Player A acts first, then Player B.
- On your turn, use the negotiate tool to choose one action:
    propose — offer a specific split of ALL resources
    accept  — agree to the most recent proposal from the other player
    reject  — decline the most recent proposal
- You may only accept if the other player has made a proposal.
- Both players can see the full negotiation history, including all proposals, actions, and messages from every prior turn.

DEADLINE:
- Rounds 1–4 are guaranteed.
- Starting in round 5, there is a 30% chance the game ends after each round with no deal. You will NOT know in advance which round is the last.
- Every round from 5 onward could be your final chance to make a deal.

SCORING:
- Deal reached: your score = sum(your_valuation x quantity_you_receive) for each resource, divided by the maximum you could score if you received everything. Range: 0.0 to 1.0.
- No deal (game ends without agreement): both players score -0.5. This is the worst possible outcome — any deal that gives you even a single point beats no deal.

RULES:
- my_share + their_share must exactly equal the pool for every resource.
- Use the negotiate tool to submit every move. Think carefully before acting."""


# -- Scenario generation -------------------------------------------------------


def _random_valuations_constrained(rng: random.Random, pool: dict) -> dict | None:
    """Generate valuations where sum(val[r] * pool[r]) == TARGET_VALUE.

    All valuations >= 1 and <= MAX_VALUATION.  Returns None if no valid
    assignment exists.
    """
    vals = {r: 1 for r in RESOURCE_TYPES}
    base_total = sum(pool[r] for r in RESOURCE_TYPES)
    remaining = TARGET_VALUE - base_total
    if remaining < 0:
        return None

    for _ in range(500):
        test = dict(vals)
        left = remaining
        resources = list(RESOURCE_TYPES)
        rng.shuffle(resources)
        for r in resources[:-1]:
            max_add = left // pool[r] if pool[r] > 0 else 0
            max_add = min(max_add, MAX_VALUATION - 1)
            if max_add > 0:
                add = rng.randint(0, max_add)
                test[r] += add
                left -= add * pool[r]
        last = resources[-1]
        if pool[last] > 0 and left % pool[last] == 0:
            test[last] += left // pool[last]
            if test[last] > MAX_VALUATION:
                continue
            total = sum(test[r] * pool[r] for r in RESOURCE_TYPES)
            if total == TARGET_VALUE:
                return test

    return None


def generate_scenario(seed: int) -> dict:
    """Generate a deterministic negotiation scenario.

    Constraints (Lewis et al., 2017):
    - sum(valuation[r] * pool[r]) == 100 for both players
    - All valuations >= 1
    - Valuations are asymmetric
    """
    rng = random.Random(seed)

    for _ in range(1000):
        pool = {r: rng.randint(3, 15) for r in RESOURCE_TYPES}
        if sum(pool[r] for r in RESOURCE_TYPES) > TARGET_VALUE:
            continue

        vals_a = _random_valuations_constrained(rng, pool)
        if vals_a is None:
            continue
        vals_b = _random_valuations_constrained(rng, pool)
        if vals_b is None:
            continue

        if vals_a != vals_b:
            return {
                "pool": pool,
                "valuations_a": vals_a,
                "valuations_b": vals_b,
                "seed": seed,
            }

    raise RuntimeError(f"Failed to generate valid scenario for seed {seed}")


# -- Validation & scoring ------------------------------------------------------


def validate_offer(offer: dict, pool: dict) -> bool:
    """Check that my_share + their_share == pool for each resource, no negatives."""
    my = offer.get("my_share", {})
    theirs = offer.get("their_share", {})
    for r in RESOURCE_TYPES:
        if my.get(r, 0) + theirs.get(r, 0) != pool[r]:
            return False
        if my.get(r, 0) < 0 or theirs.get(r, 0) < 0:
            return False
    return True


def score_split(valuations: dict, split: dict) -> int:
    return sum(valuations[r] * split.get(r, 0) for r in RESOURCE_TYPES)


def max_possible(valuations: dict, pool: dict) -> int:
    return sum(valuations[r] * pool[r] for r in RESOURCE_TYPES)


# -- Prompt building -----------------------------------------------------------


def build_system_prompt(
    role: str,
    pool: dict,
    valuations: dict,
    custom_strategy: str | None = None,
) -> str:
    pool_str = ", ".join(f"{v} {k}" for k, v in pool.items())
    val_str = ", ".join(f"{k}: {v} points each" for k, v in valuations.items())
    prompt = RULES_PREAMBLE.format(role=role, pool_str=pool_str, val_str=val_str)
    if custom_strategy:
        prompt += f"\n\nADDITIONAL STRATEGY INSTRUCTIONS:\n{custom_strategy}"
    return prompt


def build_turn_prompt(
    history: list[dict],
    role: str,
    round_number: int,
    pool: dict,
) -> str:
    parts = []

    if round_number > GUARANTEED_ROUNDS:
        parts.append(
            f"OVERTIME (round {round_number}) — The game could end after this round. "
            "If no deal is reached before the game ends, both players score -0.5."
        )

    if history:
        parts.append("NEGOTIATION HISTORY:")
        for turn in history:
            speaker = f"Player {turn['player']}"
            is_self = turn["player"] == role

            if turn["action"] == "propose" and turn.get("offer"):
                if is_self:
                    parts.append(
                        f"  {speaker} (you) proposed: you keep {turn['offer']['my_share']}, "
                        f"they get {turn['offer']['their_share']}"
                    )
                else:
                    parts.append(
                        f"  {speaker} proposed: they keep {turn['offer']['my_share']}, "
                        f"you get {turn['offer']['their_share']}"
                    )
            elif turn["action"] == "accept":
                parts.append(
                    f"  {speaker} {'(you) ' if is_self else ''}ACCEPTED the proposal"
                )
            elif turn["action"] == "reject":
                parts.append(
                    f"  {speaker} {'(you) ' if is_self else ''}rejected the proposal"
                )

            parts.append(f'    Message: "{turn["message"]}"')
        parts.append("")

    parts.append(
        f"Round {round_number} — Your turn (Player {role}). Use the negotiate tool."
    )
    return "\n".join(parts)
