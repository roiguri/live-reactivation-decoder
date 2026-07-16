from __future__ import annotations

import subprocess

import pytest

from backend.online_phase import stream_source
from backend.online_phase.stream_source import LslProxySource, StreamSource


class FakePopen:
    """Minimal subprocess.Popen stand-in that stays 'alive' until terminated."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.pid = 4321
        self._returncode = None
        self.terminated = False
        self.killed = False

    @property
    def returncode(self):
        return self._returncode

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = 0

    def kill(self):
        self.killed = True
        self._returncode = -9

    def wait(self, timeout=None):
        return self._returncode

    def communicate(self, timeout=None):
        return (b"", b"")


@pytest.fixture
def patched(monkeypatch, tmp_path):
    proxy_path = tmp_path / "LSLProxy.exe"
    proxy_path.write_bytes(b"stub")
    monkeypatch.setattr(stream_source.time, "sleep", lambda *_: None)
    created: list[FakePopen] = []

    def _factory(*args, **kwargs):
        proc = FakePopen(*args, **kwargs)
        created.append(proc)
        return proc

    monkeypatch.setattr(stream_source.subprocess, "Popen", _factory)
    # start() runs `taskkill /F /IM LSLProxy.exe` via subprocess.run to reap
    # orphaned proxies. Stub it so the test neither shells out nor routes that
    # call through the Popen factory above (which would pollute `created`).
    monkeypatch.setattr(stream_source.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(stream_source.os, "name", "nt")
    return proxy_path, created


def test_lsl_proxy_source_satisfies_protocol(patched):
    proxy_path, _ = patched
    assert isinstance(LslProxySource(proxy_path), StreamSource)


def test_start_launches_proxy_once(patched):
    proxy_path, created = patched
    source = LslProxySource(proxy_path)

    source.start()
    assert source.is_running is True
    assert len(created) == 1

    # Idempotent: already-running proxy is not relaunched.
    source.start()
    assert len(created) == 1


def test_start_raises_when_executable_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(stream_source.os, "name", "nt")
    source = LslProxySource(tmp_path / "missing.exe")
    with pytest.raises(FileNotFoundError):
        source.start()


def test_start_raises_when_proxy_dies_immediately(patched, monkeypatch):
    proxy_path, _ = patched

    class DeadPopen(FakePopen):
        def poll(self):
            return 1  # already exited

    monkeypatch.setattr(stream_source.subprocess, "Popen", DeadPopen)
    source = LslProxySource(proxy_path)
    with pytest.raises(RuntimeError, match="failed to start"):
        source.start()


def test_stop_terminates_and_is_idempotent(patched):
    proxy_path, created = patched
    source = LslProxySource(proxy_path)
    source.start()

    source.stop()
    assert created[0].terminated is True
    assert source.is_running is False

    # Idempotent when already stopped.
    source.stop()


def test_stop_kills_when_terminate_times_out(patched, monkeypatch):
    proxy_path, _ = patched

    class StubbornPopen(FakePopen):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._waits = 0

        def wait(self, timeout=None):
            self._waits += 1
            if self._waits == 1:
                raise subprocess.TimeoutExpired(cmd="proxy", timeout=timeout)
            return self._returncode

    monkeypatch.setattr(stream_source.subprocess, "Popen", StubbornPopen)
    source = LslProxySource(proxy_path)
    source.start()
    source.stop()

    assert source.proxy_process is None
