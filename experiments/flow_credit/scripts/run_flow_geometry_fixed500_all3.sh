#!/usr/bin/env bash
set -uo pipefail

PROJECT=/data/Wayne/gzw/rlinf_gr00t_n17
BULK=/mnt/models/gzw/rlinf_gr00t_n17
RLINF="$PROJECT/RLinf"

PROBE="$RLINF/experiments/flow_credit/scripts/run_flow_geometry_probe.sh"
ANALYZE="$RLINF/experiments/flow_credit/analysis/analyze_flow_geometry.py"

LABELS=(
    sft_base
    step400
    step600
)

OFFSETS=(
    0
    100
    200
    300
    400
)

NUM_ENVS=100

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="$BULK/evaluations/flow_geometry_fixed500_all3_$STAMP"

mkdir -p "$ROOT"

echo "$ROOT" \
    > "$BULK/logs/flow_geometry_fixed500_all3.latest"

echo "$(git -C "$RLINF" rev-parse HEAD)" \
    > "$ROOT/git_commit.txt"

git -C "$RLINF" status -sb \
    > "$ROOT/git_status.txt"

cp \
    "$RLINF/examples/embodiment/config/libero_spatial_n17_fixed100_eval_gpu23.yaml" \
    "$ROOT/config_snapshot.yaml"

python - <<'PY' > "$ROOT/environment.txt"
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)

try:
    import ray
    print("ray:", ray.__version__)
except Exception as exc:
    print("ray: ERROR", exc)

try:
    import rlinf
    print("rlinf:", getattr(rlinf, "__version__", "unknown"))
except Exception as exc:
    print("rlinf: ERROR", exc)

try:
    import gr00t
    print("gr00t:", getattr(gr00t, "__version__", "unknown"))
except Exception as exc:
    print("gr00t: ERROR", exc)
PY

MANIFEST="$ROOT/manifest.tsv"

printf \
"label\toffset\tnum_envs\trun_rc\tanalysis_rc\toutput_dir\tsuccess_once\treward\treturn\tepisode_len\tnum_trajectories\traw_files\traw_bytes\n" \
> "$MANIFEST"

echo "============================================================"
echo "GR00T FLOW GEOMETRY FIXED500 × ALL3"
echo "ROOT=$ROOT"
echo "Models: ${LABELS[*]}"
echo "Offsets: ${OFFSETS[*]}"
echo "Total episodes target: 1500"
echo "============================================================"

for LABEL in "${LABELS[@]}"
do
    for OFFSET in "${OFFSETS[@]}"
    do
        echo
        echo "############################################################"
        echo "MODEL=$LABEL"
        echo "OFFSET=$OFFSET"
        echo "NUM_ENVS=$NUM_ENVS"
        echo "############################################################"

        START_TIME=$(date +%s)

        set +e

        EVAL_LABEL="$LABEL" \
        bash "$PROBE" \
            "$OFFSET" \
            "$LABEL" \
            "$NUM_ENVS"

        RUN_RC=$?

        set -e

        RUN_DIR=$(
            cat "$BULK/logs/flow_geometry_probe.latest" \
            2>/dev/null || true
        )

        ANALYSIS_RC=99

        if [[ -n "$RUN_DIR" && -d "$RUN_DIR/raw" ]]
        then
            RAW_COUNT=$(
                find "$RUN_DIR/raw" \
                    -type f \
                    -name 'flow_*.npz' \
                    | wc -l
            )

            if [[ "$RAW_COUNT" -gt 0 ]]
            then
                mkdir -p "$RUN_DIR/figures"

                set +e

                python "$ANALYZE" \
                    "$RUN_DIR/raw" \
                    "$RUN_DIR/figures" \
                    > "$RUN_DIR/geometry_analysis.log" \
                    2>&1

                ANALYSIS_RC=$?

                set -e
            fi
        fi

        END_TIME=$(date +%s)

        echo "$((END_TIME - START_TIME))" \
            > "$RUN_DIR/runtime_seconds.txt" 2>/dev/null || true

        if [[ -d "$RUN_DIR" ]]
        then
            ln -s "$RUN_DIR" \
                "$ROOT/${LABEL}_offset${OFFSET}" \
                2>/dev/null || true
        fi

        python - \
            "$LABEL" \
            "$OFFSET" \
            "$NUM_ENVS" \
            "$RUN_RC" \
            "$ANALYSIS_RC" \
            "$RUN_DIR" \
            "$MANIFEST" \
            <<'PY'
import json
import sys
from pathlib import Path

(
    label,
    offset,
    num_envs,
    run_rc,
    analysis_rc,
    run_dir,
    manifest,
) = sys.argv[1:]

run = Path(run_dir)
metrics = {}

metrics_path = run / "metrics.json"

if metrics_path.exists():
    try:
        payload = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
        metrics = payload.get("metrics", {})
    except Exception:
        pass

raw_dir = run / "raw"

raw_files = 0
raw_bytes = 0

if raw_dir.exists():
    for path in raw_dir.glob("flow_*.npz"):
        raw_files += 1
        try:
            raw_bytes += path.stat().st_size
        except OSError:
            pass

values = [
    label,
    offset,
    num_envs,
    run_rc,
    analysis_rc,
    str(run),
    metrics.get("success_once", "NA"),
    metrics.get("reward", "NA"),
    metrics.get("return", "NA"),
    metrics.get("episode_len", "NA"),
    metrics.get("num_trajectories", "NA"),
    raw_files,
    raw_bytes,
]

with open(
    manifest,
    "a",
    encoding="utf-8",
) as f:
    f.write("\t".join(map(str, values)) + "\n")
PY

        echo
        echo "===== SUBRUN RESULT ====="
        tail -n 1 "$MANIFEST"

        if [[ -f "$RUN_DIR/geometry_analysis.log" ]]
        then
            echo
            cat "$RUN_DIR/geometry_analysis.log"
        fi

        # Ray 必须完全退出后再进入下一组。
        ray stop --force >/dev/null 2>&1 || true
        sleep 5
    done
done

echo
echo "============================================================"
echo "ALL REQUESTED RUNS FINISHED"
echo "ROOT=$ROOT"
echo "============================================================"

echo
echo "===== FINAL MANIFEST ====="
cat "$MANIFEST"

touch "$ROOT/FLOW_GEOMETRY_FIXED500_ALL3_COMPLETE"
