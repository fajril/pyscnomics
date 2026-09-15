from click.testing import CliRunner
from pytest import MonkeyPatch

from pyscnomics.pyscnomics_cli import entry_point


def test_root_help_lists_serve_command() -> None:
    result = CliRunner().invoke(entry_point, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output


def test_serve_uses_default_host_and_port(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, str, int, bool]] = []

    def fake_run(application: str, *, host: str, port: int, reload: bool) -> None:
        calls.append((application, host, port, reload))

    monkeypatch.setattr("pyscnomics.pyscnomics_cli.uvicorn.run", fake_run)

    result = CliRunner().invoke(entry_point, ["serve"])

    assert result.exit_code == 0
    assert calls == [("pyscnomics.api.main:app", "127.0.0.1", 9999, False)]


def test_serve_forwards_host_and_port_options(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, str, int, bool]] = []

    def fake_run(application: str, *, host: str, port: int, reload: bool) -> None:
        calls.append((application, host, port, reload))

    monkeypatch.setattr("pyscnomics.pyscnomics_cli.uvicorn.run", fake_run)

    result = CliRunner().invoke(
        entry_point,
        ["serve", "--host", "0.0.0.0", "--port", "8080"],
    )

    assert result.exit_code == 0
    assert calls == [("pyscnomics.api.main:app", "0.0.0.0", 8080, False)]


def test_legacy_api_option_is_rejected() -> None:
    result = CliRunner().invoke(entry_point, ["--api", "1"])

    assert result.exit_code != 0
    assert "No such option" in result.output
