from __future__ import annotations

from types import SimpleNamespace

from experiments.exp4_analysis.launcher import poll_derivation, submit_derivation


class FakeDeployedFunction:
    def __init__(self) -> None:
        self.spawned = []

    def spawn(self, which: str) -> SimpleNamespace:
        self.spawned.append(which)
        return SimpleNamespace(object_id="fc-derived-stable")

    def remote(self, *_args) -> None:
        raise AssertionError("durable derivation must not use remote()")


def test_submit_uses_deployed_spawn_and_returns_stable_id(capsys) -> None:
    deployed = FakeDeployedFunction()
    lookups = []

    class Function:
        @staticmethod
        def from_name(app_name: str, function_name: str) -> FakeDeployedFunction:
            lookups.append((app_name, function_name))
            return deployed

    call_id = submit_derivation("main", SimpleNamespace(Function=Function))
    assert call_id == "fc-derived-stable"
    assert capsys.readouterr().out == "fc-derived-stable\n"
    assert lookups == [("vmp-exp4-derive", "derive")]
    assert deployed.spawned == ["main"]


def test_poll_reattaches_without_cancelling() -> None:
    observed = {}

    class Call:
        def get(self, *, timeout: float) -> None:
            observed["timeout"] = timeout
            raise TimeoutError

        def cancel(self) -> None:
            observed["cancelled"] = True

    class FunctionCall:
        @staticmethod
        def from_id(call_id: str) -> Call:
            observed["call_id"] = call_id
            return Call()

    modal = SimpleNamespace(FunctionCall=FunctionCall, exception=SimpleNamespace())
    assert poll_derivation("fc-existing", modal_module=modal) == {
        "call_id": "fc-existing",
        "status": "running",
    }
    assert observed == {"call_id": "fc-existing", "timeout": 0.0}
