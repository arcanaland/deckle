#!/usr/bin/env bash
# Pull the most recent scan(s) off Dropbox — the fallback acquisition path for scans taken
# away from this machine, via the printer's "Save to Dropbox" preset. Scans taken here go
# over eSCL instead and never touch Dropbox.
#
#   tools/pull-scan.sh                 # newest scan into ./inbox
#   tools/pull-scan.sh -n 4            # four newest
#   tools/pull-scan.sh -o /tmp/x       # somewhere else
#   deckle detect "$(tools/pull-scan.sh)"
#
# Local paths go to stdout, one per line, and nothing else does — so the last form works.
# The remote holds unrelated PDFs alongside the scans, so only image extensions are
# considered. Files already present locally are not re-fetched but are still printed.

set -euo pipefail

remote=${DECKLE_SCAN_REMOTE:-dropbox:Inbox}
dest=inbox
count=1

usage() { sed -n '2,14p' "$0" | cut -c3-; exit "${1:-0}"; }

while getopts ':n:o:r:h' opt; do
  case $opt in
    n) count=$OPTARG ;;
    o) dest=$OPTARG ;;
    r) remote=$OPTARG ;;
    h) usage ;;
    *) usage 2 ;;
  esac
done

[[ $count =~ ^[1-9][0-9]*$ ]] || { echo "pull-scan: -n wants a positive integer, got '$count'" >&2; exit 2; }

# `tp` is modtime then path; ISO timestamps sort lexically, so reverse sort is newest-first.
# Sorting on the remote's modtime rather than on the timestamp-shaped filename means a scan
# renamed by hand still lands in the right place.
names=$(
  rclone lsf --files-only --format tp --separator $'\t' "$remote" \
    | grep -Ei '\.(jpe?g|png|tiff?)$' \
    | sort -r \
    | head -n "$count" \
    | cut -f2-
)

[[ -n $names ]] || { echo "pull-scan: no image files under $remote" >&2; exit 1; }

mkdir -p "$dest"
while IFS= read -r name; do
  if [[ ! -f "$dest/$name" ]]; then
    rclone copyto "$remote/$name" "$dest/$name" >&2
    echo "pull-scan: fetched $name" >&2
  else
    echo "pull-scan: $name already local" >&2
  fi
  printf '%s\n' "$dest/$name"
done <<<"$names"
