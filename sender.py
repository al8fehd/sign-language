from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CommandSender:
    mode: str  # "off" | "udp" | "tcp"
    host: str
    port: int
    min_interval_ms: int = 120
    send_only_on_change: bool = True

    _last_sent_cmd: Optional[str] = None
    _last_sent_at: float = 0.0
    _tcp_sock: Optional[socket.socket] = None

    def close(self) -> None:
        if self._tcp_sock is not None:
            try:
                self._tcp_sock.close()
            finally:
                self._tcp_sock = None

    def _should_send(self, cmd: str) -> bool:
        now = time.time()
        if self.send_only_on_change and (cmd == self._last_sent_cmd):
            if (now - self._last_sent_at) * 1000.0 < self.min_interval_ms:
                return False
        if (now - self._last_sent_at) * 1000.0 < self.min_interval_ms:
            return False
        return True

    def send(self, cmd: str) -> None:
        if self.mode == "off":
            return
        if not self._should_send(cmd):
            return

        payload = (cmd.strip() + "\n").encode("ascii", "ignore")

        if self.mode == "udp":
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(payload, (self.host, self.port))
        elif self.mode == "tcp":
            if self._tcp_sock is None:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.5)
                s.connect((self.host, self.port))
                s.settimeout(None)
                self._tcp_sock = s
            self._tcp_sock.sendall(payload)
        else:
            raise ValueError(f"Unknown send mode: {self.mode}")

        self._last_sent_cmd = cmd
        self._last_sent_at = time.time()

