from __future__ import annotations

from types import SimpleNamespace

from experiments.exp4_collection.launcher import poll_collection, submit_collection


class FakeDeployedFunction:
    def __init__(self) -> None:
        self.spawned: list[tuple[str, str]] = []

    def spawn(self, which: str, dataset_variant: str) -> SimpleNamespace:
        self.spawned.append((which, dataset_variant))
        return SimpleNamespace(object_id="fc-stable-123")

    def remote(self, *_args: object) -> None:
        raise AssertionError("the launcher must never use remote()")


def test_submit_looks_up_deployment_and_prints_stable_spawn_id(capsys) -> None:
    deployed = FakeDeployedFunction()
    lookups: list[tuple[str, str]] = []

    class Function:
        @staticmethod
        def from_name(app_name: str, function_name: str) -> FakeDeployedFunction:
            lookups.append((app_name, function_name))
            return deployed

    modal = SimpleNamespace(Function=Function)

    call_id = submit_collection("main", modal)

    assert call_id == "fc-stable-123"
    assert capsys.readouterr().out == "fc-stable-123\n"
    assert lookups == [("vmp-exp4-collect", "collect")]
    assert deployed.spawned == [("main", "run_v1")]


def test_submit_can_select_leak_free_dataset(capsys) -> None:
    deployed = FakeDeployedFunction()

    class Function:
        @staticmethod
        def from_name(_app_name: str, _function_name: str) -> FakeDeployedFunction:
            return deployed

    modal = SimpleNamespace(Function=Function)

    submit_collection("lbr", modal, dataset_variant="leak_free")

    assert capsys.readouterr().out == "fc-stable-123\n"
    assert deployed.spawned == [("lbr", "leak_free")]


def test_poll_reattaches_without_cancelling_prior_call() -> None:
    observed: dict[str, object] = {}

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

    result = poll_collection("fc-prior-456", modal_module=modal)

    assert result == {"call_id": "fc-prior-456", "status": "running"}
    assert observed == {"call_id": "fc-prior-456", "timeout": 0.0}
