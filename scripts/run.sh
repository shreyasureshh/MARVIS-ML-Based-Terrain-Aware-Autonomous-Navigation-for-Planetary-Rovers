#!/usr/bin/env bash
# MARVIS pipeline runner. One stage per invocation:
#   scripts/run.sh data            # inspect + split (needs data/raw/ai4mars or MARVIS_RAW_ROOT)
#   scripts/run.sh smoke           # real-data smoke test: 1 epoch on 64 images (run dir tagged _smoke)
#   scripts/run.sh bench           # time one real training epoch of E1 (tagged _bench)
#   scripts/run.sh train E1 [seed] # train one experiment (E1..E5)
#   scripts/run.sh eval <run>      # val+test evaluation, threshold calibration on val, planning on test
#   scripts/run.sh all [seeds...]  # E1..E5 x seeds, then eval + compare   (default seed 0)
#   scripts/run.sh compare         # aggregate runs, select demo model by val mIoU
#   scripts/run.sh dashboard
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export PYTHONPATH=src
cfg() { ls configs/experiments/"$1"_*.yaml | head -1; }
name() { basename "$(cfg "$1")" .yaml; }
case "${1:-}" in
  data)    $PY -m marvis.prepare inspect && $PY -m marvis.prepare split ;;
  smoke)   $PY -m marvis.train "$(cfg E1)" --epochs 1 --max-train-images 64 --tag smoke --set max_val_images=32
           $PY -m marvis.evaluate "$(name E1)_s0_smoke" --split val --mc 5 ;;
  bench)   $PY -m marvis.train "$(cfg E1)" --epochs 1 --tag bench ;;
  train)   $PY -m marvis.train "$(cfg "$2")" --seed "${3:-0}" --resume ;;
  eval)    $PY -m marvis.evaluate "$2" --split val --save-preds
           $PY -m marvis.evaluate "$2" --split test --save-preds
           for m in data/manifests/test_*.csv; do [ -e "$m" ] && $PY -m marvis.evaluate "$2" --split "$(basename "$m" .csv)"; done
           $PY -m marvis.evaluate_planning "$2" --split val --calibrate
           $PY -m marvis.evaluate_planning "$2" --split test --n 60 ;;
  all)     shift; seeds=("${@:-0}")
           for s in "${seeds[@]}"; do for e in E1 E2 E3 E4 E5; do
             $PY -m marvis.train "$(cfg $e)" --seed "$s" --resume
             "$0" eval "$(name $e)_s$s"
           done; done
           $PY -m marvis.compare ;;
  compare) $PY -m marvis.compare ;;
  dashboard) exec .venv/bin/streamlit run app/dashboard.py ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
