from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from arxiv_cslg_search.data.models import ArxivPaper

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
OPENSEARCH_NS = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
ARXIV_API_URL = "https://export.arxiv.org/api/query"


@dataclass(frozen=True)
class DateWindow:
    label: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class ArxivQueryPage:
    total_results: int
    start_index: int
    items_per_page: int
    papers: tuple[ArxivPaper, ...]


def build_search_query(category: str, start_date: str, end_date: str) -> str:
    start_ts = start_date.replace("-", "") + "0000"
    end_ts = end_date.replace("-", "") + "2359"
    return f"cat:{category} AND submittedDate:[{start_ts} TO {end_ts}]"


def parse_feed(xml_text: str) -> ArxivQueryPage:
    root = ET.fromstring(xml_text)
    total_results = int(root.findtext("opensearch:totalResults", default="0", namespaces=OPENSEARCH_NS))
    start_index = int(root.findtext("opensearch:startIndex", default="0", namespaces=OPENSEARCH_NS))
    items_per_page = int(root.findtext("opensearch:itemsPerPage", default="0", namespaces=OPENSEARCH_NS))

    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = _require_text(entry, "atom:id")
        title = _normalize_text(_require_text(entry, "atom:title"))
        abstract = _normalize_text(_require_text(entry, "atom:summary"))
        updated = _require_text(entry, "atom:updated")
        published = _require_text(entry, "atom:published")
        authors = tuple(
            _normalize_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
            if author.findtext("atom:name", default="", namespaces=ATOM_NS)
        )
        categories = tuple(
            category.attrib["term"]
            for category in entry.findall("atom:category", ATOM_NS)
            if "term" in category.attrib
        )
        primary_category = categories[0] if categories else ""
        pdf_url = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break
        papers.append(
            ArxivPaper(
                arxiv_id=entry_id.rsplit("/", 1)[-1],
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                primary_category=primary_category,
                published=published,
                updated=updated,
                abs_url=entry_id.replace("http://", "https://"),
                pdf_url=pdf_url,
            )
        )

    return ArxivQueryPage(
        total_results=total_results,
        start_index=start_index,
        items_per_page=items_per_page,
        papers=tuple(papers),
    )


class ArxivClient:
    def __init__(self, *, page_size: int = 100, delay_seconds: float = 3.0) -> None:
        self.page_size = page_size
        self.delay_seconds = delay_seconds
        self._last_request_at: float | None = None

    def fetch_papers(
        self,
        *,
        search_query: str,
        limit: int,
        raw_dir: Path | None = None,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> list[ArxivPaper]:
        papers: list[ArxivPaper] = []
        start = 0
        page_number = 0
        while len(papers) < limit:
            page_limit = min(self.page_size, limit - len(papers))
            xml_text = self._request(search_query, start, page_limit, sort_by, sort_order)
            if raw_dir is not None:
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f"page-{page_number:03d}.xml"
                raw_path.write_text(xml_text, encoding="utf-8")
            page = parse_feed(xml_text)
            if not page.papers:
                break
            papers.extend(page.papers)
            start += len(page.papers)
            page_number += 1
            if start >= page.total_results:
                break
        return papers[:limit]

    def _request(
        self,
        search_query: str,
        start: int,
        max_results: int,
        sort_by: str,
        sort_order: str,
    ) -> str:
        if self._last_request_at is not None and self.delay_seconds > 0:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self.delay_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)

        params = urllib.parse.urlencode(
            {
                "search_query": search_query,
                "start": start,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }
        )
        url = f"{ARXIV_API_URL}?{params}"
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read().decode("utf-8")
        self._last_request_at = time.monotonic()
        return body


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _require_text(entry: ET.Element, path: str) -> str:
    text = entry.findtext(path, default="", namespaces=ATOM_NS)
    if not text:
        raise ValueError(f"Missing required field {path}")
    return text
