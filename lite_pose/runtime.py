"""Shared runtime conventions for command-line model workflows."""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
from typing import Any, Callable, Optional, TypeVar, Union

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_F = TypeVar("_F", bound=Callable[..., Any])


def configure_logging(level: Optional[Union[str, int]] = None) -> None:
    """Configure deterministic, process-wide console logging."""
    selected = level or os.getenv("MODEL_LOG_LEVEL", "INFO")
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=selected,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout,
    )


def command(name: str) -> Callable[[_F], _F]:
    """Add consistent lifecycle logging and error handling to a CLI command."""
    def decorate(function: _F) -> _F:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            configure_logging()
            logger = logging.getLogger(name)
            started = time.monotonic()
            logger.info("event=started command=%s framework=pytorch", name)
            try:
                result = function(*args, **kwargs)
            except KeyboardInterrupt:
                logger.warning("event=interrupted command=%s", name)
                raise SystemExit(130) from None
            except Exception:
                logger.exception("event=failed command=%s", name)
                raise
            logger.info(
                "event=completed command=%s elapsed_seconds=%.3f",
                name,
                time.monotonic() - started,
            )
            return result

        return wrapped  # type: ignore[return-value]

    return decorate
