#!/usr/bin/env python3
"""Compare champion and challenger checkpoints on the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.metrics import evaluate_checkpoint
from training.utils import load_config, resolve_path, setup_logging

logger = setup_logging(__name__)


def compare_models(
    champion_path: Path,
    challenger_path: Path,
    config_path: Path | None = None,
    *,
    min_improvement: float = 0.0,
) -> dict:
    """Evaluate both checkpoints and decide whether the challenger should replace the champion."""
    champion = evaluate_checkpoint(champion_path, config_path)
    challenger = evaluate_checkpoint(challenger_path, config_path)

    margin = challenger["test_f1_macro"] - champion["test_f1_macro"]
    if margin > min_improvement:
        winner = "challenger"
    elif margin < -min_improvement:
        winner = "champion"
    else:
        winner = "champion"

    return {
        "winner": winner,
        "margin_f1_macro": margin,
        "min_improvement": min_improvement,
        "champion": champion,
        "challenger": challenger,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two model checkpoints on the test set.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--champion", type=Path, default=None, help="Current production checkpoint")
    parser.add_argument("--challenger", type=Path, required=True, help="Candidate checkpoint")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=None,
        help="Minimum F1 improvement required for challenger to win",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    config = load_config(args.config)
    retrain_cfg = config.get("retraining", {})
    champion = args.champion or resolve_path(retrain_cfg.get("champion_checkpoint", "models/checkpoints/best_model.pt"))
    min_improvement = (
        args.min_improvement
        if args.min_improvement is not None
        else float(retrain_cfg.get("min_improvement", 0.0))
    )
    output = args.output or resolve_path(
        retrain_cfg.get("comparison_output", "models/evaluation/comparison.json")
    )

    result = compare_models(champion, args.challenger, args.config, min_improvement=min_improvement)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    logger.info("Champion test F1:   %.4f", result["champion"]["test_f1_macro"])
    logger.info("Challenger test F1: %.4f", result["challenger"]["test_f1_macro"])
    logger.info("Margin: %.4f (min required: %.4f)", result["margin_f1_macro"], min_improvement)
    logger.info("Winner: %s", result["winner"])
    logger.info("Comparison saved to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
