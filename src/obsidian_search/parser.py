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
    tags: list[str]
    aliases: list[str]
    frontmatter: dict


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

    # Prepend title to content for better embedding context
    embedding_content = f"{title}\n\n{clean_content}"

    return ParsedNote(
        path=str(file_path),
        title=title,
        content=embedding_content,
        tags=tags,
        aliases=aliases,
        frontmatter=frontmatter,
    )
