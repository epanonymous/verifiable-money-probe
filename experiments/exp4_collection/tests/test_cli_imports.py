from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def test_poll_dispatches_without_importing_numpy() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import json
        import sys
        import types

        class BlockNumpy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numpy" or fullname.startswith("numpy."):
                    raise AssertionError("NumPy import attempted")
                return None

        sys.meta_path.insert(0, BlockNumpy())
        launcher = types.ModuleType("experiments.exp4_collection.launcher")
        launcher.poll_collection = lambda call_id, timeout: {
            "call_id": call_id,
            "status": "mocked",
            "timeout": timeout,
        }
        sys.modules[launcher.__name__] = launcher

        from experiments.exp4_collection.__main__ import main

        raise SystemExit(main(["poll", "fc-test", "--timeout", "1.5"]))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "call_id": "fc-test",
        "status": "mocked",
        "timeout": 1.5,
    }


def test_submit_dispatches_leak_free_dataset_variant() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys
        import types

        observed = {}
        launcher = types.ModuleType("experiments.exp4_collection.launcher")
        def submit(which, *, dataset_variant):
            observed.update(which=which, dataset_variant=dataset_variant)
        launcher.submit_collection = submit
        sys.modules[launcher.__name__] = launcher

        from experiments.exp4_collection.__main__ import main

        status = main(["submit", "main", "--leak-free"])
        print(json.dumps({"status": status, **observed}, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "status": 0,
        "which": "main",
        "dataset_variant": "leak_free",
    }
