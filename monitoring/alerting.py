"""Alert hooks for future integration with Slack, PagerDuty, or email."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from monitoring.drift_check import DriftAlert

AlertHook = Callable[[DriftAlert, dict[str, Any]], None]
logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self) -> None:
        self._hooks: list[AlertHook] = []

    def register_hook(self, hook: AlertHook) -> None:
        self._hooks.append(hook)

    def notify(self, alert: DriftAlert, context: dict[str, Any]) -> None:
        logger.warning("Monitoring alert [%s]: %s", alert.code, alert.message)
        for hook in self._hooks:
            hook(alert, context)
