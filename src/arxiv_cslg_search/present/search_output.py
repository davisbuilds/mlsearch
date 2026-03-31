from __future__ import annotations

import json

from arxiv_cslg_search.retrieval.search import SearchHit


def render_hits(hits: list[SearchHit], *, output_format: str) -> str:
    if output_format == "json":
        return json.dumps([hit.__dict__ for hit in hits], indent=2, sort_keys=True)

    lines = []
    for index, hit in enumerate(hits, start=1):
        published = hit.published[:10]
        lines.append(f"{index:>2}. {hit.arxiv_id}  {hit.score:.4f}  {published}  {hit.title}")
    return "\n".join(lines)
