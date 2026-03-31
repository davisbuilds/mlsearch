from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QueryCandidate:
    query_id: str
    query_text: str
    style: str
    source_paper_id: str
    source_title: str
    source_published: str
    positive_ids: tuple[str, ...]
    hard_negative_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["positive_ids"] = list(self.positive_ids)
        payload["hard_negative_ids"] = list(self.hard_negative_ids)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryCandidate":
        return cls(
            query_id=data["query_id"],
            query_text=data["query_text"],
            style=data["style"],
            source_paper_id=data["source_paper_id"],
            source_title=data["source_title"],
            source_published=data["source_published"],
            positive_ids=tuple(data["positive_ids"]),
            hard_negative_ids=tuple(data["hard_negative_ids"]),
        )
