from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    primary_category: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["authors"] = list(self.authors)
        payload["categories"] = list(self.categories)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArxivPaper":
        return cls(
            arxiv_id=data["arxiv_id"],
            title=data["title"],
            abstract=data["abstract"],
            authors=tuple(data["authors"]),
            categories=tuple(data["categories"]),
            primary_category=data["primary_category"],
            published=data["published"],
            updated=data["updated"],
            abs_url=data["abs_url"],
            pdf_url=data.get("pdf_url"),
        )
