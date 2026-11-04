from pathlib import Path

from bs4 import BeautifulSoup

from dcp.extract import register
from dcp.extract.text import read_text
from dcp.models import TextBlock

_DROP = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"]
_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote", "pre", "dd", "dt"]


def html_to_blocks(markup: str, location_prefix: str = "") -> list[TextBlock]:
    soup = BeautifulSoup(markup, "lxml")
    for tag in soup(_DROP):
        tag.decompose()
    blocks = []
    for el in soup.find_all(_BLOCK_TAGS):
        # Only leaf-level blocks, so nested <li><p> isn't emitted twice.
        if el.find(_BLOCK_TAGS):
            continue
        text = el.get_text(" ", strip=True)
        if text:
            blocks.append(TextBlock(text, f"{location_prefix}{el.name}"))
    if not blocks and soup.body:
        blocks.append(TextBlock(soup.body.get_text("\n", strip=True), location_prefix))
    return blocks


@register("html", ".html", ".htm", ".xhtml")
def extract_html(path: Path) -> list[TextBlock]:
    return html_to_blocks(read_text(path))


@register("epub", ".epub")
def extract_epub(path: Path) -> list[TextBlock]:
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path))
    blocks = []
    for i, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), 1):
        blocks.extend(html_to_blocks(item.get_content(), f"chapter={i}:"))
    return blocks
