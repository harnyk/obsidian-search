# Sync All

Bump version, reinstall tool, and push changes to git.

## Steps

1. Bump the patch version in `pyproject.toml`
2. Force reinstall the tool: `uv tool install . --force`
3. Git add all changes
4. Git commit with a descriptive message
5. Git push to remote
