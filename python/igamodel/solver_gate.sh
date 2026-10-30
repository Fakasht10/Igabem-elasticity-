#!/usr/bin/env bash
# Solver-level round trip: bc_case on a case file as written by hand (A) and as
# read into the model tree and written back (B) must give byte-identical CSV and VTU.
# usage: python/igamodel/solver_gate.sh <build>/bc_case cases/*.json
set -euo pipefail
bc_case=$(realpath "$1"); shift
here=$(cd "$(dirname "$0")/.." && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
for c in "$@"; do
  dir=$(dirname "$(realpath "$c")")
  python3 - "$c" "$dir" "$here" <<'PY'
import pathlib, sys
c, d, here = sys.argv[1:]
sys.path.insert(0, here)
import igamodel as im
t = pathlib.Path(c).read_text()
i = t.index('{\n') + 2                      # same case, operator cache off, as written by hand
pathlib.Path(d, '_gateA.json').write_text(t[:i] + '  "cache": "none",\n' + t[i:])
m = im.read_case(c)
m.apply({"op": "set", "tag": "root", "key": "cache", "value": "none"})
im.write_case(m, pathlib.Path(d, '_gateB.json'))
PY
  "$bc_case" "$dir/_gateA.json" --csv "$out/A.csv" --vtu "$out/A" > "$out/A.log" 2>&1
  "$bc_case" "$dir/_gateB.json" --csv "$out/B.csv" --vtu "$out/B" > "$out/B.log" 2>&1
  rm -f "$dir/_gateA.json" "$dir/_gateB.json"
  same=yes
  cmp -s "$out/A.csv" "$out/B.csv" || same=no
  for f in "$out"/A*.vtu; do cmp -s "$f" "${f/\/A/\/B}" || same=no; done
  echo "$(basename "$c"): identical=$same ($(tail -1 "$out/A.log" | sed 's/# //'))"
  [ $same = yes ] || exit 1
  rm -f "$out"/*
done
