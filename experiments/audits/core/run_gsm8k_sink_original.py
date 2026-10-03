"""
Run the existing ORIGINAL matched-control GSM8K runner,
but replace all-agent final aggregation with sink-only aggregation.

All CLI arguments are forwarded to:
    experiments.audits.run_gsm8k_matched_control
"""

import runpy

from experiments.audits.decision_policy_patch import install_sink_only_policy


if __name__ == "__main__":
    install_sink_only_policy()

    runpy.run_module(
        "experiments.audits.run_gsm8k_matched_control",
        run_name="__main__",
    )
