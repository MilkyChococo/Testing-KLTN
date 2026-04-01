from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GraphRelation:
    source_id: str
    target_id: str
    relation: str
    score: float = 1.0
