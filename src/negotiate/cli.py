"""CLI for the negotiation challenge test harness."""

import asyncio
import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import RESOURCE_TYPES, RULES_PREAMBLE, HARD_MAX_ROUNDS

console = Console()


def _format_offer(offer: dict, perspective: str = "proposer") -> str:
    """Format an offer for display."""
    my = offer["my_share"]
    theirs = offer["their_share"]
    my_str = " ".join(f"{my[r]}{r[0]}" for r in RESOURCE_TYPES)
    their_str = " ".join(f"{theirs[r]}{r[0]}" for r in RESOURCE_TYPES)
    if perspective == "proposer":
        return f"keep [{my_str}] give [{their_str}]"
    return f"get [{their_str}] give [{my_str}]"


def _action_style(action: str) -> str:
    if action == "propose":
        return "cyan"
    if action == "accept":
        return "green"
    return "red"


def _render_game(game: dict, index: int, show_reasoning: bool = False):
    """Render a single game's turn-by-turn history."""
    pool = game["pool"]
    vals_u = game["valuations_user"]
    vals_b = game["valuations_baseline"]
    pool_str = ", ".join(f"{v} {k}" for k, v in pool.items())
    val_u_str = ", ".join(f"{k}={v}" for k, v in vals_u.items())
    val_b_str = ", ".join(f"{k}={v}" for k, v in vals_b.items())

    user_role = game.get("user_role", "A")
    outcome = "[green]DEAL[/green]" if game["deal_reached"] else "[red]NO DEAL[/red]"
    header = (
        f"Game {index + 1} (You={user_role})  |  {outcome} round {game['final_round']}  |  "
        f"You: {game['user_score']:.2f}  Baseline: {game['baseline_score']:.2f}\n"
        f"Pool: {pool_str}\n"
        f"Your vals: {val_u_str}  |  Baseline vals: {val_b_str}"
    )

    user_role = game.get("user_role", "A")

    lines = []
    for turn in game["turns"]:
        is_user = turn["player"] == user_role
        player_label = f"You ({turn['player']})" if is_user else f"Opp ({turn['player']})"
        style = _action_style(turn["action"])

        line = f"  R{turn['round']} [{style}]{player_label} {turn['action'].upper()}[/{style}]"
        if turn["action"] == "propose" and turn.get("offer"):
            line += f"  {_format_offer(turn['offer'])}"
        if turn.get("message"):
            msg = turn["message"][:80]
            line += f'\n       "{msg}"'
        if show_reasoning and turn.get("reasoning"):
            reasoning = turn["reasoning"][:200]
            line += f"\n       [dim]Thinking: {reasoning}[/dim]"
        lines.append(line)

    body = "\n".join(lines)
    console.print(Panel(body, title=header, border_style="dim", expand=False))


def _render_summary(result: dict):
    """Render the summary stats table."""
    stats = result["stats"]

    table = Table(title="Evaluation Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Mean Score", f"{stats['mean']:.4f}")
    table.add_row("Median Score", f"{stats['median']:.4f}")
    table.add_row("Min / Max", f"{stats['min']:.4f} / {stats['max']:.4f}")
    table.add_row("Std Dev", f"{stats['std']:.4f}")
    table.add_row("Deal Rate", f"{stats['deals_reached']}/{stats['games_played']} ({stats['deal_rate']:.0%})")
    table.add_row("Elapsed", f"{result['elapsed']:.1f}s")
    table.add_row("Seed", str(result["seed"]))

    console.print()
    console.print(table)


@click.group()
def cli():
    """Negotiation challenge test harness."""
    pass


@cli.command()
@click.argument("prompt", type=click.Path(exists=True))
@click.option("-n", "--games", default=10, show_default=True, help="Number of games to play.")
@click.option("-s", "--seed", type=int, default=None, help="Base seed for reproducibility.")
@click.option("-v", "--verbose", is_flag=True, help="Show turn-by-turn for every game.")
@click.option("--reasoning", is_flag=True, help="Show model thinking (implies --verbose).")
@click.option("--save", "save_path", type=click.Path(), default=None, help="Save full results to JSON.")
@click.option("--concurrency", default=20, show_default=True, help="Max concurrent API calls.")
def test(prompt, games, seed, verbose, reasoning, save_path, concurrency):
    """Run your strategy prompt against the baseline.

    PROMPT is a path to a text file containing your strategy (max 2000 chars).
    """
    prompt_text = Path(prompt).read_text().strip()
    if not prompt_text:
        console.print("[red]Error:[/red] Prompt file is empty.")
        sys.exit(1)
    if len(prompt_text) > 2000:
        console.print(f"[red]Error:[/red] Prompt is {len(prompt_text)} chars (max 2000).")
        sys.exit(1)

    console.print(f"[bold]Strategy:[/bold] {Path(prompt).name} ({len(prompt_text)} chars)")
    console.print(f"[bold]Games:[/bold] {games}  [bold]Seed:[/bold] {seed or 'random'}")
    console.print()

    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        console.print(
            "[red]Error:[/red] Set GOOGLE_API_KEY (or GEMINI_API_KEY) env var.\n"
            "  Get one at https://ai.google.dev/gemini-api/docs/api-key"
        )
        sys.exit(1)

    if reasoning:
        verbose = True

    from .runner import run_evaluation

    def on_progress(completed, total, game_result):
        score = game_result["user_score"]
        deal = "deal" if game_result["deal_reached"] else "no deal"
        role = game_result.get("user_role", "A")
        console.print(
            f"  [{completed}/{total}] score={score:.2f} ({deal}, role={role})",
            highlight=False,
        )

    result = asyncio.run(
        run_evaluation(
            prompt=prompt_text,
            num_games=games,
            seed=seed,
            concurrency=concurrency,
            on_game_complete=on_progress,
        )
    )

    # Show game details
    if verbose:
        console.print()
        # Sort games by scenario seed for stable display order
        sorted_games = sorted(result["games"], key=lambda g: g["scenario_seed"])
        for i, game in enumerate(sorted_games):
            _render_game(game, i, show_reasoning=reasoning)

    _render_summary(result)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(json.dumps(result, indent=2))
        console.print(f"\nResults saved to {save_path}")


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--game", "game_index", type=int, default=None, help="Show only game N (0-indexed).")
@click.option("--reasoning", is_flag=True, help="Show model thinking.")
def inspect(results_file, game_index, reasoning):
    """Inspect saved results from a previous test run.

    RESULTS_FILE is the JSON file saved with `negotiate test --save`.
    """
    result = json.loads(Path(results_file).read_text())

    console.print(f"[bold]Prompt:[/bold] {result['prompt'][:80]}...")
    console.print()

    games = sorted(result["games"], key=lambda g: g["scenario_seed"])

    if game_index is not None:
        if game_index < 0 or game_index >= len(games):
            console.print(f"[red]Error:[/red] Game index {game_index} out of range (0-{len(games)-1}).")
            sys.exit(1)
        _render_game(games[game_index], game_index, show_reasoning=reasoning)
    else:
        for i, game in enumerate(games):
            _render_game(game, i, show_reasoning=reasoning)

    _render_summary(result)


@cli.command()
def rules():
    """Print the game rules preamble."""
    console.print(
        Panel(
            RULES_PREAMBLE.replace("{role}", "_").replace("{pool_str}", "...").replace("{val_str}", "..."),
            title="Game Rules",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    cli()
