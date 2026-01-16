"""Markdown and frontmatter parsing utilities."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ParsedNote:
    """Parsed markdown note with extracted metadata."""

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

    remaining = content[match.end() :]
    return frontmatter, remaining


def extract_title(frontmatter: dict, content: str, file_path: Path) -> str:
    """Extract title from frontmatter, first heading, or filename."""
    # Try frontmatter title
    if frontmatter.get("title"):
        return frontmatter["title"]

    # Try first H1 heading
    h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    # Fall back to filename without extension
    return file_path.stem


def clean_content_for_embedding(content: str) -> str:
    """Clean markdown content for better embedding quality."""
    # Remove code blocks (keep the text but remove the markers)
    content = re.sub(r"```[\w]*\n?", "", content)

    # Remove inline code backticks
    content = re.sub(r"`([^`]+)`", r"\1", content)

    # Remove wiki-style links but keep the display text
    # [[link|display]] -> display, [[link]] -> link
    content = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", content)
    content = re.sub(r"\[\[([^\]]+)\]\]", r"\1", content)

    # Remove markdown links but keep the display text
    # [display](url) -> display
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

    # Remove images
    content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", content)

    # Remove HTML tags
    content = re.sub(r"<[^>]+>", "", content)

    # Remove heading markers but keep text
    content = re.sub(r"^#{1,6}\s+", "", content, flags=re.MULTILINE)

    # Remove horizontal rules
    content = re.sub(r"^[-*_]{3,}\s*$", "", content, flags=re.MULTILINE)

    # Remove emphasis markers but keep text
    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
    content = re.sub(r"\*([^*]+)\*", r"\1", content)
    content = re.sub(r"__([^_]+)__", r"\1", content)
    content = re.sub(r"_([^_]+)_", r"\1", content)

    # Remove blockquote markers
    content = re.sub(r"^>\s*", "", content, flags=re.MULTILINE)

    # Collapse multiple newlines
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Strip whitespace
    content = content.strip()

    return content


def chunk_text(text: str, title: str) -> list[str]:
    """Split text into overlapping chunks for better embedding.

    Each chunk is prefixed with the title for context.
    """
    if len(text) <= CHUNK_SIZE:
        return [f"{title}\n\n{text}"] if text.strip() else []

    chunks = []
    # Split into paragraphs first
    paragraphs = re.split(r"\n\n+", text)

    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(para) + 2 > CHUNK_SIZE:
            if current_chunk:
                chunks.append(f"{title}\n\n{current_chunk.strip()}")

            # If paragraph itself is too long, split it
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
                # Start new chunk with overlap from previous
                if chunks:
                    # Get last ~CHUNK_OVERLAP chars from previous chunk content
                    prev_content = chunks[-1][len(title) + 2:]  # Remove title prefix
                    overlap = prev_content[-CHUNK_OVERLAP:].lstrip()
                    # Find word boundary
                    space_idx = overlap.find(" ")
                    if space_idx > 0:
                        overlap = overlap[space_idx + 1:]
                    current_chunk = f"{overlap}\n\n{para}" if overlap else para
                else:
                    current_chunk = para
        else:
            current_chunk = f"{current_chunk}\n\n{para}".strip()

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(f"{title}\n\n{current_chunk.strip()}")

    return chunks


def parse_note(file_path: Path) -> ParsedNote:
    """Parse a markdown note file."""
    content = file_path.read_text(encoding="utf-8")
    frontmatter, body = extract_frontmatter(content)

    title = extract_title(frontmatter, body, file_path)

    # Extract tags from frontmatter
    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]

    # Extract aliases from frontmatter
    aliases = frontmatter.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]

    # Clean content for embedding
    clean_content = clean_content_for_embedding(body)

    # Create chunks for embedding
    chunks = chunk_text(clean_content, title)

    # Full content for storage
    full_content = f"{title}\n\n{clean_content}"

    return ParsedNote(
        path=str(file_path),
        title=title,
        content=full_content,
        chunks=chunks,
        tags=tags,
        aliases=aliases,
        frontmatter=frontmatter,
    )
