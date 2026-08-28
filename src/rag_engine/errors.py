"""Exception types shared across the package.

Kept in their own module so any module can raise them without importing a
sibling that imports it back.
"""

from __future__ import annotations


class ConfigError(ValueError):
    """A setting is missing, unparseable, or outside the range the engine accepts.

    Subclasses :class:`ValueError` so callers that catch the broader type keep
    working. The distinct type exists so a caller can separate "this is
    configured wrongly" from "this run failed", which deserve different
    handling: the first is fixed by editing the environment, the second is not.
    """
