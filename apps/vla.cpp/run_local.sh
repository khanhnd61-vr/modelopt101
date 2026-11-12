#!/usr/bin/env bash
# Run SmolVLA on LIBERO locally (Linux / WSL2 + NVIDIA), no notebook.
# Assumes vla.cpp is built, the bundle downloaded and the LIBERO env set up (see README.md).
#
#   bash run_local.sh one      # one libero_object episode (task 0)
#   bash run_local.sh all      # all 10 libero_object tasks, one episode each
#   bash run_local.sh multi    # pick 3-4 objects in a row in one scene (run_multi_pick.py)
#
# From Windows (VS Code terminal): wsl bash apps/vla.cpp/run_local.sh one
#
# Env: N=10 episodes per task, TASKS="0 5" task ids for `all`, SERVER_ARGS="--weight-dtype f16" extra vla-server flags,
#      MODEL=path to a (quantized) gguf, default models/smolvla/smolvla-libero.gguf,
#      OUT=dir for videos and logs (default ~/libero_outputs).
set -euo pipefail

MODE=${1:-one}
VLA_CPP=${VLA_CPP:-$HOME/vla.cpp}
OUT=${OUT:-$HOME/libero_outputs}  # not /tmp: WSL wipes it when the VM idles out
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$VLA_CPP/eval/sim/libero/libero_uv/.venv/bin/python"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl VLA_CPP

cd "$VLA_CPP"
LOG="$OUT/vla-server.log"
mkdir -p "$OUT"
./build/vla-server ${SERVER_ARGS:-} --bind tcp://127.0.0.1:5555 "${MODEL:-models/smolvla/smolvla-libero.gguf}" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 600); do
    grep -qE "bound to .* ready" "$LOG" && break
    kill -0 $SERVER 2>/dev/null || { tail -50 "$LOG"; exit 1; }
    sleep 1
done
grep -E "backend =|weight_buf" "$LOG" || true

episode() {  # $1 = task id
    "$PY" eval/client/run_sim_client_direct.py \
        --task libero_object --task-id "$1" --n-episodes "${N:-1}" \
        --output-dir "$OUT" --arch smolvla --n-action-steps 10
}

case $MODE in
    one)
        episode 0
        cat "$OUT/smolvla/libero_object/task_0/summary.txt"
        ;;
    all)
        for t in ${TASKS:-$(seq 0 9)}; do
            episode "$t" > "$OUT/task_$t.log" 2>&1 || echo "task $t failed, see $OUT/task_$t.log"
            echo "task $t: $(grep -oE "'task': '[^']+'" "$OUT/task_$t.log" | head -1)" \
                 "$(grep -hE 'Success rate|per step' "$OUT/smolvla/libero_object/task_$t/summary.txt" | tr '\n' ' ')"
        done
        ;;
    multi)
        "$PY" "$HERE/run_multi_pick.py" --output-dir "$OUT/multi_pick" "${@:2}"
        ;;
    *)
        echo "usage: $0 one|all|multi [run_multi_pick.py args]"; exit 1
        ;;
esac
echo "videos: $OUT"
