#!/usr/bin/env bash
# Reorder the queue: stop the running one-seed job and the three waiting jobs, then start the
# chain  seed0 -> extras -> big -> after.  The hold file keeps the Mac-side script from
# stopping the instance before the last job (after) has ended.
set -uo pipefail
W=/workspace/process-not-obey
cd $W
touch .hold_stop
OLD=$(tmux list-panes -t seed0 -F "#{pane_pid}" 2>/dev/null | head -1)
echo "old one-seed job: session leader pid ${OLD:-none}"
for s in after extras big seed0; do tmux kill-session -t $s 2>/dev/null && echo "killed tmux session $s"; done
if [ -n "${OLD:-}" ]; then
  for i in 1 2 3 4 5 6 7 8; do
    left=$(pgrep -s "$OLD" | tr "\n" " " || true)
    [ -z "$left" ] && break
    echo "still alive from the old job: $left"
    if [ $i -ge 4 ]; then kill -KILL $left 2>/dev/null; else kill -TERM $left 2>/dev/null; fi
    sleep 5
  done
  left=$(pgrep -s "$OLD" | tr "\n" " " || true)
  [ -n "$left" ] && { echo "ERROR: processes of the old job are still alive: $left; nothing started"; exit 1; }
fi
if [ -f .seed0_done ]; then echo "removing stray marker: $(cat .seed0_done)"; rm -f .seed0_done; fi
for m in .after_done .extras_done .big_done; do [ -f $m ] && { echo "ERROR: unexpected marker $m; nothing started"; exit 1; }; done
for i in 1 2 3 4 5 6; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && break; sleep 5
done
echo "GPU memory in use: ${used} MiB"
[ "$used" -lt 2000 ] || { echo "ERROR: GPU memory was not released; nothing started"; exit 1; }
echo; echo "=================== $(date -u +%H:%M:%S) queue reordered: SecFid of the six runs first (run_seed0b.sh) ===================" >> $W/seed0.log
tmux new-session -d -s seed0 "(bash /workspace/run_seed0b.sh >> $W/seed0.log 2>&1; echo exit=\$? > $W/.seed0_done)"
tmux new-session -d -s extras "cd $W; while [ ! -f .seed0_done ]; do sleep 30; done; (bash /workspace/run_extras.sh > extras.log 2>&1; echo exit=\$? > .extras_done)"
tmux new-session -d -s big "cd $W; while [ ! -f .extras_done ]; do sleep 30; done; (bash /workspace/run_c1_variant.sh secbig 7500 3 1.2756 > big.log 2>&1; echo exit=\$? > .big_done)"
tmux new-session -d -s after "cd $W; while [ ! -f .big_done ]; do sleep 30; done; (bash /workspace/run_after.sh > after.log 2>&1; echo exit=\$? > .after_done); rm -f .hold_stop"
sleep 3
tmux ls
tmux list-panes -a -F "#{session_name}: #{pane_start_command}" | grep -v "^main\|^ssh_tmux" | cut -c1-230
