#!/bin/bash
# usage: tools/make_zip.sh upload/v2lr  -> upload/v2lr.zip with agent.py at the top level
set -e; d=$1; rm -f "$d.zip"; (cd "$d" && zip -q -r "../$(basename $d).zip" . -x '__pycache__/*'); unzip -l "$d.zip"
