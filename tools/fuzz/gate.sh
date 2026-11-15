#!/bin/bash
# gate.sh BASE_BIN CAND_BIN [OUTDIR]
#   No-panic, no-new-divergence A/B for the merge gate. BASE_BIN/CAND_BIN are built `conformance` test
#   binaries (`env -u D2D_FONTS cargo test -p d2d --test conformance --no-run` prints the path). Every
#   corpus listed in the corpus file runs through the `fuzz_diff` test for both binaries, in parallel.
#
#   Exit 0 (GATE PASS): the candidate has no panic and no seed that diverges in CAND but matched in BASE.
#   Exit 1 (GATE FAIL): a candidate panic or a new divergence (listed). Seeds the candidate FIXES are
#   reported, not gated.
#
# Corpora come from a config file, one `name:dir:start:count:browser` per line (`#` comments allowed),
# because the pinned corpora are private and live outside the repo:
#   GATE_CORPORA   path to that file (default: tools/fuzz/gate.corpora next to this script)
#   GATE_ACCEPTED  optional "accepted trade" file, lines "<corpus> <seed> <reason>": such a seed is
#                  reported but not gated (a divergence the owner has signed off on, pending a fix).
# `dir` may use $VARS (e.g. a $FUZZ root); they are expanded against the environment.
set -u
BASE=$1; CAND=$2; O=${3:-$(mktemp -d)}; mkdir -p "$O"
HERE=$(cd "$(dirname "$0")" && pwd)
CORPORA_FILE=${GATE_CORPORA:-$HERE/gate.corpora}
ACCEPTED=${GATE_ACCEPTED:-}
[ -f "$CORPORA_FILE" ] || { echo "no corpus file: $CORPORA_FILE (copy gate.corpora.example and edit)" >&2; exit 2; }

# Load corpora, skipping blanks/comments and expanding $VARS in the dir field.
CORPORA=()
while IFS= read -r line; do
  line=${line%%#*}; line=$(echo "$line" | xargs); [ -z "$line" ] && continue
  IFS=: read -r name dir start n br <<< "$line"
  dir=$(eval echo "$dir")
  CORPORA+=("$name:$dir:$start:$n:$br")
done < "$CORPORA_FILE"

run() { # name tag bin dir start n browser → $O/<name>-<tag>.{seeds,skips,rc,log}
  local name=$1 tag=$2 bin=$3 dir=$4 start=$5 n=$6 br=$7
  FUZZ_BROWSER=$br FUZZ_DIFF_DIR=$dir FUZZ_START=$start FUZZ_N=$n timeout 1800 \
    env -u D2D_FONTS "$bin" fuzz_diff --nocapture > "$O/$name-$tag.log" 2>&1
  echo $? > "$O/$name-$tag.rc"
  grep -E "^FDIFF " "$O/$name-$tag.log" | sed 's/^FDIFF \([0-9]*\):.*/\1/' | sort -u > "$O/$name-$tag.seeds"
  grep -E "^FDIFF-SKIP " "$O/$name-$tag.log" | sed 's/^FDIFF-SKIP \([0-9]*\):.*/\1/' | sort -u > "$O/$name-$tag.skips"
}

for c in "${CORPORA[@]}"; do
  IFS=: read -r name dir start n br <<< "$c"
  [ -d "$dir" ] || { echo "missing corpus $dir" >&2; exit 2; }
  run "$name" base "$BASE" "$dir" "$start" "$n" "$br" &
  run "$name" cand "$CAND" "$dir" "$start" "$n" "$br" &
done
wait

fail=0
for c in "${CORPORA[@]}"; do
  IFS=: read -r name _ <<< "$c"
  # A seed base SKIPPED (not comparable there) that cand now compares and finds different is newly
  # comparable, not a regression: report it, do not gate on it.
  newcmp=$(comm -13 "$O/$name-base.seeds" "$O/$name-cand.seeds" | comm -12 - "$O/$name-base.skips" | tr '\n' ' ')
  new=$(comm -13 "$O/$name-base.seeds" "$O/$name-cand.seeds" | comm -23 - "$O/$name-base.skips" | tr '\n' ' ')
  # Owner-approved trades: reported, not gated. Each stays routed to an owner; remove its line once fixed.
  if [ -n "$ACCEPTED" ] && [ -f "$ACCEPTED" ] && [ -n "$new" ]; then
    acc=$(awk -v c="$name" '$1==c {print $2}' "$ACCEPTED" | sort -u)
    if [ -n "$acc" ]; then
      accepted=$(printf '%s\n' $new | sort -u | comm -12 - <(printf '%s\n' $acc) | tr '\n' ' ')
      new=$(printf '%s\n' $new | sort -u | comm -23 - <(printf '%s\n' $acc) | tr '\n' ' ' | sed 's/^ *$//')
      [ -n "$accepted" ] && echo "         accepted trade (not gated): $accepted"
    fi
  fi
  fixed=$(comm -23 "$O/$name-base.seeds" "$O/$name-cand.seeds" | tr '\n' ' ')
  panic=$(grep -m1 "panicked at" "$O/$name-cand.log")
  bpanic=$(grep -c "panicked at" "$O/$name-base.log")  # a base panic hides the rest of its chunk
  printf '%-8s base %4s cand %4s  fixed: %s new: %s\n' "$name" \
    "$(grep -c . "$O/$name-base.seeds")" "$(grep -c . "$O/$name-cand.seeds")" "$fixed" "$new"
  [ -n "$newcmp" ] && echo "         newly comparable (base skipped, not gated): $newcmp"
  [ "$bpanic" -gt 0 ] && echo "         (base panicked in this chunk: its later seeds are unchecked)"
  if [ -n "$panic" ]; then echo "  PANIC in candidate: $panic"; fail=1; fi
  if ! grep -q "^fuzz_diff: ok=" "$O/$name-cand.log"; then echo "  candidate did not finish (rc $(cat "$O/$name-cand.rc"))"; fail=1; fi
  [ -n "$new" ] && fail=1
done
echo "logs: $O"
[ $fail = 0 ] && echo "GATE PASS" || echo "GATE FAIL"
exit $fail
