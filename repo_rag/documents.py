from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Document:
    """Lightweight representation of a chunkable file."""

    path: str
    content: str
    metadata: Optional[Dict[str, str]] = None

    def as_metadata(self) -> Dict[str, str]:
        base_metadata: Dict[str, str] = {"path": self.path}
        if self.metadata:
            base_metadata.update(self.metadata)
        return base_metadata

