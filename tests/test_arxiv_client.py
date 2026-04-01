from __future__ import annotations

from mlsearch.data.arxiv_client import build_search_query, parse_feed


SAMPLE_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>2</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/1604.00092v1</id>
    <updated>2016-04-01T01:04:31Z</updated>
    <published>2016-04-01T01:04:31Z</published>
    <title>Variational reaction-diffusion systems for semantic segmentation</title>
    <summary> A  sample abstract with
      line breaks. </summary>
    <author><name>Alice Example</name></author>
    <author><name>Bob Example</name></author>
    <link href="https://arxiv.org/abs/1604.00092v1" rel="alternate" type="text/html" />
    <link href="https://arxiv.org/pdf/1604.00092v1" rel="related" type="application/pdf" title="pdf" />
    <category term="cs.LG" />
    <category term="cs.CV" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1604.00093v1</id>
    <updated>2016-04-01T01:05:00Z</updated>
    <published>2016-04-01T01:05:00Z</published>
    <title>Another paper</title>
    <summary>Another abstract.</summary>
    <author><name>Carol Example</name></author>
    <link href="https://arxiv.org/abs/1604.00093v1" rel="alternate" type="text/html" />
    <category term="cs.LG" />
  </entry>
</feed>
"""


def test_build_search_query_uses_category_and_date_range() -> None:
    query = build_search_query("cs.LG", "2016-04-01", "2017-03-31")
    assert query == "cat:cs.LG AND submittedDate:[201604010000 TO 201703312359]"


def test_parse_feed_extracts_normalized_papers() -> None:
    page = parse_feed(SAMPLE_FEED)
    assert page.total_results == 2
    assert page.items_per_page == 2
    assert len(page.papers) == 2

    first = page.papers[0]
    assert first.arxiv_id == "1604.00092v1"
    assert first.abstract == "A sample abstract with line breaks."
    assert first.authors == ("Alice Example", "Bob Example")
    assert first.categories == ("cs.LG", "cs.CV")
    assert first.pdf_url == "https://arxiv.org/pdf/1604.00092v1"
