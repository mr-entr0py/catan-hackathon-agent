#!/bin/bash
# usage: tools/sweep.sh GAMES SEED0 'json1' 'json2' ...   (runs sequentially, each on all cores)
G=$1; S=$2; shift 2
for c in "$@"; do
  CATAN_CFG="$c" python tools/peval.py --a agents/v2/agent.py --b examples/f_agent.py --games $G --seed0 $S --jobs 4 --tag "v2 $c" | sed "s|^|$c  |"
done
