#!/bin/bash

# Fail fast if anything fails
set -e

# Extract version from pyproject.toml using grep and sed
VERSION=$(grep -m 1 '^version' pyproject.toml | sed -E 's/version *= *"([^"]+)"/\1/')

# Format tag
TAG="v$VERSION"

# Confirm with user
echo "Tagging current commit as $TAG and pushing to origin"
read -p "Continue? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" ]]; then
    echo "Aborted."
    exit 1
fi

# Create and push the tag
git tag "$TAG"
git push origin "$TAG"

echo "✅ Tag $TAG created and pushed."
