from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def sandbox(enabled: bool = True):
    """Placeholder for sandbox: isolates tool execution when enabled.

    Currently a no-op context manager; replace with real sandboxing as needed.
    """
    try:
        yield
    finally:
        pass

