from __future__ import annotations

import subprocess
import sys
import textwrap


def test_submit_dispatches_without_importing_numpy() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import types

        class BlockNumpy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numpy" or fullname.startswith("numpy."):
                    raise AssertionError("NumPy import attempted")
                return None

        sys.meta_path.insert(0, BlockNumpy())
        launcher = types.ModuleType("experiments.exp4_analysis.launcher")
        launcher.submit_derivation = lambda which: print("mocked:" + which)
        sys.modules[launcher.__name__] = launcher

        from experiments.exp4_analysis.__main__ import main

        raise SystemExit(main(["submit", "main"]))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "mocked:main\n"
