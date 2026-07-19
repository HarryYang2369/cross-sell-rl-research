"""End-to-end: the CLI runs a small experiment and writes its outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from cross_sell_rl.run import main


def test_cli_end_to_end(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "results"
    config = {
        "synthetic": {"n_customers": 300, "seed": 1},
        "experiment": {
            "n_rounds": 400,
            "seed": 2,
            "output_dir": str(output_dir),
            "agents": {"random": {}, "linucb": {"alpha": 1.0}},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    main(["--config", str(config_path)])

    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert set(metrics["agent"]) == {"random", "linucb"}
    assert (output_dir / "learning_curves.png").exists()
    printed = capsys.readouterr().out
    assert "Summary" in printed
