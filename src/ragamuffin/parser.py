"""Document parsing utilities for markdown, plain text, and RST files."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ParsedDocument:
    """Parsed document with extracted metadata."""

    path: str
    title: str
    content: str
    chunks: list[str]
    tags: list[str]
    aliases: list[str]
    frontmatter: dict


# Chunking parameters
CHUNK_SIZE = 1500  # characters (~375 tokens)
CHUNK_OVERLAP = 200  # characters overlap between chunks

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# RST patterns to strip
_RST_DIRECTIVE = re.compile(r"^\.\. [\w:-]+::.*$", re.MULTILINE)
_RST_ROLE = re.compile(r":[a-zA-Z][\w-]*:`[^`]*`")
_RST_HYPERLINK_TARGET = re.compile(r"^\.\. _[^:]+:.*$", re.MULTILINE)
_RST_ANONYMOUS_HYPERLINK = re.compile(r"__ http\S+")
_RST_FIELD_LIST = re.compile(r"^:[^:]+:.*$", re.MULTILINE)
_RST_LITERAL_BLOCK_MARKER = re.compile(r"::\s*$", re.MULTILINE)
_RST_SECTION_UNDERLINE = re.compile(r"^[=\-~^\"'`#*+]{3,}\s*$", re.MULTILINE)


# ============================================================================
# Shared helpers
# ============================================================================


def chunk_text(text: str, title: str) -> list[str]:
    """Split text into overlapping chunks for better embedding.

    Each chunk is prefixed with the title for context.
    """
    if len(text) <= CHUNK_SIZE:
        return [f"{title}\n\n{text}"] if text.strip() else []

    chunks = []
    paragraphs = re.split(r"\n\n+", text)

    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 > CHUNK_SIZE:
            if current_chunk:
                chunks.append(f"{title}\n\n{current_chunk.strip()}")

            if len(para) > CHUNK_SIZE:
                words = para.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 > CHUNK_SIZE:
                        if current_chunk:
                            chunks.append(f"{title}\n\n{current_chunk.strip()}")
                        current_chunk = word
                    else:
                        current_chunk = f"{current_chunk} {word}".strip()
            else:
                if chunks:
                    prev_content = chunks[-1][len(title) + 2:]
                    overlap = prev_content[-CHUNK_OVERLAP:].lstrip()
                    space_idx = overlap.find(" ")
                    if space_idx > 0:
                        overlap = overlap[space_idx + 1:]
                    current_chunk = f"{overlap}\n\n{para}" if overlap else para
                else:
                    current_chunk = para
        else:
            current_chunk = f"{current_chunk}\n\n{para}".strip()

    if current_chunk.strip():
        chunks.append(f"{title}\n\n{current_chunk.strip()}")

    return chunks


# ============================================================================
# Markdown parser
# ============================================================================


def extract_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown content.

    Returns (frontmatter_dict, remaining_content).
    """
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content

    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        frontmatter = {}

    remaining = content[match.end():]
    return frontmatter, remaining


def extract_title(frontmatter: dict, content: str, file_path: Path) -> str:
    """Extract title from frontmatter, first heading, or filename."""
    if frontmatter.get("title"):
        return frontmatter["title"]

    h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    return file_path.stem


def clean_markdown_for_embedding(content: str) -> str:
    """Clean markdown content for better embedding quality."""
    content = re.sub(r"```[\w]*\n?", "", content)
    content = re.sub(r"`([^`]+)`", r"\1", content)
    content = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", content)
    content = re.sub(r"\[\[([^\]]+)\]\]", r"\1", content)
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
    content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", content)
    content = re.sub(r"<[^>]+>", "", content)
    content = re.sub(r"^#{1,6}\s+", "", content, flags=re.MULTILINE)
    content = re.sub(r"^[-*_]{3,}\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
    content = re.sub(r"\*([^*]+)\*", r"\1", content)
    content = re.sub(r"__([^_]+)__", r"\1", content)
    content = re.sub(r"_([^_]+)_", r"\1", content)
    content = re.sub(r"^>\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def parse_markdown(file_path: Path) -> ParsedDocument:
    """Parse a markdown (.md) file."""
    content = file_path.read_text(encoding="utf-8")
    frontmatter, body = extract_frontmatter(content)

    title = extract_title(frontmatter, body, file_path)

    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]

    aliases = frontmatter.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]

    clean_content = clean_markdown_for_embedding(body)
    chunks = chunk_text(clean_content, title)
    full_content = f"{title}\n\n{clean_content}"

    return ParsedDocument(
        path=str(file_path),
        title=title,
        content=full_content,
        chunks=chunks,
        tags=tags,
        aliases=aliases,
        frontmatter=frontmatter,
    )


# ============================================================================
# Plain text parser
# ============================================================================


def parse_txt(file_path: Path) -> ParsedDocument:
    """Parse a plain text (.txt) file."""
    content = file_path.read_text(encoding="utf-8")
    title = file_path.stem
    clean_content = re.sub(r"\n{3,}", "\n\n", content).strip()
    chunks = chunk_text(clean_content, title)

    return ParsedDocument(
        path=str(file_path),
        title=title,
        content=f"{title}\n\n{clean_content}",
        chunks=chunks,
        tags=[],
        aliases=[],
        frontmatter={},
    )


# ============================================================================
# RST parser
# ============================================================================


def clean_rst_for_embedding(content: str) -> str:
    """Strip RST markup for better embedding quality."""
    content = _RST_DIRECTIVE.sub("", content)
    content = _RST_HYPERLINK_TARGET.sub("", content)
    content = _RST_ANONYMOUS_HYPERLINK.sub("", content)
    content = _RST_ROLE.sub(lambda m: m.group(0).split("`")[1], content)
    content = _RST_FIELD_LIST.sub("", content)
    content = _RST_LITERAL_BLOCK_MARKER.sub("", content)
    content = _RST_SECTION_UNDERLINE.sub("", content)
    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
    content = re.sub(r"\*([^*]+)\*", r"\1", content)
    content = re.sub(r"``([^`]+)``", r"\1", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _extract_rst_title(content: str, file_path: Path) -> str:
    """Extract title from the first RST section heading or filename."""
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.strip() and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line and len(next_line) >= len(line.strip()) and re.match(r"^[=\-~^\"'`#*+]+$", next_line.strip()):
                return line.strip()
    return file_path.stem


def parse_rst(file_path: Path) -> ParsedDocument:
    """Parse a reStructuredText (.rst) file."""
    content = file_path.read_text(encoding="utf-8")
    title = _extract_rst_title(content, file_path)
    clean_content = clean_rst_for_embedding(content)
    chunks = chunk_text(clean_content, title)

    return ParsedDocument(
        path=str(file_path),
        title=title,
        content=f"{title}\n\n{clean_content}",
        chunks=chunks,
        tags=[],
        aliases=[],
        frontmatter={},
    )


# ============================================================================
# Dispatcher
# ============================================================================

_PARSERS = {
    ".md": parse_markdown,
    ".txt": parse_txt,
    ".rst": parse_rst,
}

SUPPORTED_EXTENSIONS = frozenset(_PARSERS)


def parse_document(file_path: Path) -> ParsedDocument:
    """Parse a document file, dispatching by extension.

    Supports: .md, .txt, .rst
    """
    parser = _PARSERS.get(file_path.suffix.lower())
    if parser is None:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return parser(file_path)
