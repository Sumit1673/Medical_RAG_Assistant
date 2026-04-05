"""Advanced RAG Assistant Package."""

from __future__ import annotations

__version__ = "1.0.0"

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)
