"""
Day 28 — Full Pipeline Runner.

Re-run the whole pipeline from scratch to confirm reproducibility.
This script orchestrates all steps in order.
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import subprocess
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run_step(name, script_path):
    """Run a pipeline step and report success/failure."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {name}")
    logger.info(f"Script: {script_path}")
    logger.info(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        logger.info(f"\n  [PASS] {name} completed in {elapsed:.1f}s")
    else:
        logger.info(f"\n  [FAIL] {name} FAILED (exit code {result.returncode})")
        return False
    return True


def main():
    logger.info("=" * 60)
    logger.info("FRAUD RISK AGENT — FULL PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Python: {PYTHON}")

    steps = [
        ("Day 1: Load & Inspect Data", "notebooks/01_load_and_inspect.py"),
        ("Day 2: EDA", "notebooks/02_eda.py"),
        ("Day 4: Build All Features", "features/build_features.py"),
        ("Day 5: Rule Engine Baseline", "model/rule_engine.py"),
        ("Day 6: Train LightGBM", "model/train.py"),
        ("Day 7: Calibrate Model", "model/calibrate.py"),
        ("Day 8: Cost Model", "model/cost_model.py"),
        ("Day 9: Expected Loss Unit Tests", "agent/expected_loss.py"),
        ("Day 10: Decision Agent", "agent/decision_agent.py"),
        ("Days 12-13: SHAP Explainability", "model/explain.py"),
        ("Day 15: Chronological Backtest", "model/evaluate.py"),
        ("Day 16: Net Financial Impact", "model/net_financial_impact.py"),
        ("Day 17: Error Analysis", "notebooks/17_error_analysis.py"),
    ]

    total_start = time.time()
    results = []

    for name, script in steps:
        script_path = os.path.join(PROJECT_ROOT, script)
        if not os.path.exists(script_path):
            logger.info(f"\n  SKIP: {script} not found")
            results.append((name, "SKIPPED"))
            continue

        success = run_step(name, script_path)
        results.append((name, "PASS" if success else "FAIL"))

        if not success:
            logger.info(f"\nPipeline stopped at: {name}")
            break

    total_elapsed = time.time() - total_start

    # --- Summary ---
    logger.info(f"\n\n{'='*60}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*60}")
    for name, status in results:
        icon = "[PASS]" if status == "PASS" else ("[SKIP]" if status == "SKIPPED" else "[FAIL]")
        logger.info(f"  {icon} {name}: {status}")

    logger.info(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    failed = sum(1 for _, s in results if s == "FAIL")
    if failed == 0:
        logger.info("\n  [PASS] ALL STEPS PASSED -- pipeline is reproducible!")
    else:
        logger.info(f"\n  [FAIL] {failed} step(s) failed.")

    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
