#!/usr/bin/env bash
#
# get_nestgpu.sh - clones NEST GPU at the commit this package was made for.
#
# Result: <work>/src is a clean copy of NEST GPU (this folder is the "NEST GPU root":
# it has CMakeLists.txt and src/). <work> is $NESTGPU_WORK, or <package>/work.
# Everything the other scripts write goes into <work>; nothing outside it is touched.
#
# Needs git and network access to GitHub.
#
# Usage:  bash get_nestgpu.sh

set -euo pipefail

PACKAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${NESTGPU_WORK:-$PACKAGE/work}
REPO=${NESTGPU_REPO:-https://github.com/nest/nest-gpu.git}

# nest/nest-gpu, merge of pull request #143 (2026-10-02, version 2.0-100)
COMMIT=ac212d5444ac4b6f48fb9d751d8eb44340d75f33

if [ -e "$WORK/src" ]; then
    echo "$WORK/src already exists. To start again from a clean copy: rm -rf $WORK/src" >&2
    exit 1
fi

mkdir -p "$WORK"
git clone "$REPO" "$WORK/src"
git -C "$WORK/src" checkout --quiet --detach "$COMMIT"

got=$(git -C "$WORK/src" rev-parse HEAD)
if [ "$got" != "$COMMIT" ]; then
    echo "Wrong commit: $got (expected $COMMIT)" >&2
    exit 1
fi
echo "NEST GPU $COMMIT is in $WORK/src"
echo "Next: python3 $PACKAGE/scripts/add_markers.py $WORK/src"
