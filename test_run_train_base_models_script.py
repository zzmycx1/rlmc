import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
RUNNER = REPO_ROOT / "run_train_base_models.sh"


class RunTrainBaseModelsScriptTest(unittest.TestCase):
    def test_dry_run_emits_commands_for_all_supported_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "TIMESTAMP": "20260101_000000",
                    "LOG_ROOT": str(tmp_path / "runner_logs"),
                    "FEATURES": "MS",
                    "PRED_LEN": "48",
                    "SEQ_LEN": "96",
                    "MODEL": "Transformer",
                    "TRAIN_EPOCHS": "1",
                    "BATCH_SIZE": "4",
                    "NUM_WORKERS": "0",
                    "PYTHON_BIN": "python3",
                }
            )

            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity"]:
                self.assertIn(f"--dataset {dataset}", result.stdout)

            self.assertEqual(result.stdout.count("1_train_base_models.py"), 5)
            self.assertIn("--features MS", result.stdout)
            self.assertIn("--pred_len 48", result.stdout)
            self.assertIn("--model Transformer", result.stdout)
            self.assertIn("--train_epochs 1", result.stdout)
            self.assertIn("--batch_size 4", result.stdout)
            self.assertIn("--num_workers 0", result.stdout)
            self.assertTrue((tmp_path / "runner_logs" / "20260101_000000").is_dir())


if __name__ == "__main__":
    unittest.main()
