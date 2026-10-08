#!/bin/bash
# Build strixite's release container image (deploy/container/Containerfile) from a git ref of a strixite clone and
# the pinned TheRock tarball, on a Strix Halo machine (a release image is built where it is tested).
#
#   build-image.sh STRIXITE_CLONE REF [THEROCK_TARBALL]
#     STRIXITE_CLONE   a clone of github.com/shawnshekari/strixite (clean: no local changes to tracked files)
#     REF              the tag or commit to build (e.g. v0.2.0); the image is tagged strixite:<REF>
#     THEROCK_TARBALL  default ~/tools/therock-tarball/therock-dist-linux-gfx1151-10.1.0a20260822.tar.gz; its
#                      .sha256 file next to it must match (the toolchain every release so far was tested with)
#
# The build context is a fresh temporary directory holding only the source at REF (git archive - nothing untracked
# leaks in) and the tarball, so two builds from the same inputs see the same files. Prints the image's size and the
# libraries it carries at the end.
set -eo pipefail
clone=${1:?usage: build-image.sh STRIXITE_CLONE REF [THEROCK_TARBALL]}
ref=${2:?usage: build-image.sh STRIXITE_CLONE REF [THEROCK_TARBALL]}
tarball=${3:-$HOME/tools/therock-tarball/therock-dist-linux-gfx1151-10.1.0a20260822.tar.gz}
here=$(cd "$(dirname "$0")" && pwd)

command -v podman > /dev/null || { echo "build-image: podman not on PATH"; exit 1; }
git -C "$clone" rev-parse --git-dir > /dev/null 2>&1 || { echo "build-image: $clone is not a git clone"; exit 1; }
commit=$(git -C "$clone" rev-parse --verify --short "$ref^{commit}" 2> /dev/null) \
    || { echo "build-image: '$ref' is not a commit or tag in $clone"; exit 1; }
[ -f "$tarball" ] || { echo "build-image: TheRock tarball $tarball missing"; exit 1; }
[ -f "$tarball.sha256" ] || { echo "build-image: $tarball.sha256 missing - can't check the toolchain"; exit 1; }
want=$(awk '{print $1}' "$tarball.sha256")
have=$(sha256sum "$tarball" | awk '{print $1}')
[ "$want" = "$have" ] || { echo "build-image: $tarball sha256 $have, expected $want"; exit 1; }
name=$(basename "$tarball" .tar.gz)
case "$name" in therock-dist-linux-gfx1151-*) ;; *) echo "build-image: $name isn't a gfx1151 TheRock build"; exit 1 ;; esac

ctx=$(mktemp -d "${TMPDIR:-/tmp}/strixite-image.XXXXXX")
trap 'rm -rf "$ctx"' EXIT
mkdir "$ctx/src"
git -C "$clone" archive "$ref" | tar -x -C "$ctx/src"
cp "$tarball" "$ctx/"
cp "$here/Containerfile" "$ctx/"
cp "$here/NOTICE-image.md" "$ctx/"
echo "build-image: strixite $ref ($commit), toolchain $name, context $ctx"

tag="strixite:$ref"
podman build --build-arg THEROCK="$name" --label org.opencontainers.image.revision="$commit" \
    --label org.opencontainers.image.version="$ref" --label org.opencontainers.image.source=https://github.com/shawnshekari/strixite \
    --label org.opencontainers.image.title=strixite --label org.opencontainers.image.licenses=AGPL-3.0-only \
    --label org.opencontainers.image.description="strixite: Qwen3.8-Flash-Next inference on AMD Strix Halo (gfx1151)" \
    --label org.opencontainers.image.url=https://github.com/shawnshekari/strixite --label org.opencontainers.image.vendor= \
    --label name=strixite --label license=AGPL-3.0-only --label vendor= --label version="$ref" \
    --label org.opencontainers.image.name=strixite --label org.opencontainers.image.license=AGPL-3.0-only \
    -t "$tag" -f "$ctx/Containerfile" "$ctx"
echo "build-image: built $tag - $(podman image inspect "$tag" --format '{{.Size}}' | numfmt --to=iec) - libraries:"
podman run --rm --entrypoint /usr/bin/ls "$tag" /opt/strixite/lib
