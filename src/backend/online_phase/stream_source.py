from __future__ import annotations

import atexit
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


logger = logging.getLogger(__name__)


def default_proxy_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / "lslproxy" / "LSLProxy.exe"


@runtime_checkable
class StreamSource(Protocol):
    """Something that publishes an LSL stream onto the network.

    Both the NeurOne hardware bridge (``LslProxySource``) and the offline
    recording replay (``ReplaySource``) implement this protocol so the
    ``LSLReceiver`` can stay a pure consumer. ``AppSession`` owns the active
    source's lifetime; the per-run ``LiveStreamSession`` only consumes it.
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...

    @property
    def is_running(self) -> bool: ...


class LslProxySource:
    """Manage the ``LSLProxy.exe`` subprocess that bridges NeurOne to LSL.

    Windows-only: the bundled proxy is a ``.exe``. ``start()`` is idempotent —
    it will not relaunch a process that is already alive, which is what lets
    discovery and the subsequent live run share one proxy without churning the
    amplifier connection.
    """

    def __init__(self, proxy_path: str | Path | None = None) -> None:
        self.proxy_path = Path(proxy_path) if proxy_path is not None else default_proxy_path()
        self.proxy_process: Optional[subprocess.Popen] = None

    @property
    def is_running(self) -> bool:
        return self.proxy_process is not None and self.proxy_process.poll() is None

    def start(self) -> None:
        if self.is_running:
            logger.debug("LSL proxy already running, skipping launch")
            return

        if not self.proxy_path.exists():
            raise FileNotFoundError(f"LSL proxy executable not found: {self.proxy_path}")

        if os.name != "nt" and self.proxy_path.suffix.lower() == ".exe":
            raise RuntimeError(
                f"Proxy executable {self.proxy_path.name} requires Windows. "
                "Run this on the decoding machine."
            )

        # Kill any orphaned instances from a previous crashed session.
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/IM", self.proxy_path.name],
                capture_output=True,
            )

        logger.info(f"Starting LSL proxy: {self.proxy_path}")
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE: real console, no visible window
        self.proxy_process = subprocess.Popen(
            [str(self.proxy_path)],
            cwd=str(self.proxy_path.parent),
            startupinfo=startupinfo,
        )
        logger.debug(f"LSL proxy spawned with PID {self.proxy_process.pid}")
        atexit.register(self.stop)

        # Wait briefly for proxy to initialize and check if it died immediately.
        time.sleep(0.5)
        if self.proxy_process.poll() is not None:
            raise RuntimeError(
                f"LSL proxy executable failed to start. Exit code: {self.proxy_process.returncode}."
            )

    def stop(self) -> None:
        if self.proxy_process is None:
            return

        if self.proxy_process.poll() is None:
            logger.debug("Terminating LSL proxy process")
            self.proxy_process.terminate()
            try:
                self.proxy_process.wait(timeout=2.0)
                logger.debug("LSL proxy terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("LSL proxy did not terminate gracefully, killing process")
                self.proxy_process.kill()
                self.proxy_process.wait(timeout=2.0)
        self.proxy_process = None
