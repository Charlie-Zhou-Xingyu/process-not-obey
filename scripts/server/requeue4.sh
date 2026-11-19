#!/usr/bin/env bash
# Insert the two mixed conditions:  seed0 -> nocount -> sft -> mix -> extras -> big -> after.
# Only the waiting `extras` session is replaced; nothing that is running is touched.
set -uo pipefail
W=/workspace/process-not-obey
cd $W
[ -f .sft_done ] && { echo "ERROR: sft already ended; extras may have started. Nothing changed"; exit 1; }
[ -f extras.log ] && { echo "ERROR: extras already started; nothing changed"; exit 1; }
[ -f .hold_stop ] || { echo "ERROR: hold file missing; nothing changed"; exit 1; }
tmux has-session -t mix 2>/dev/null && { echo "ERROR: a mix session exists already; nothing changed"; exit 1; }
[ -f /workspace/pno-mix/data/generated/seed0/C8_mix_fidelity.jsonl ] || { echo "ERROR: the mix copy is not prepared"; exit 1; }
tmux kill-session -t extras && echo "replaced the waiting extras session"
tmux new-session -d -s mix "cd $W; while [ ! -f .sft_done ]; do sleep 30; done; (bash /workspace/run_cfg_variant.sh mix 'secfid secfid_clean sep conflict' C7_mix_ordinary C8_mix_fidelity > mix.log 2>&1; echo exit=\$? > .mix_done)"
tmux new-session -d -s extras "cd $W; while [ ! -f .mix_done ]; do sleep 30; done; (bash /workspace/run_extras.sh > extras.log 2>&1; echo exit=\$? > .extras_done)"
sleep 2
tmux ls | cut -c1-40
for s in sft mix extras; do P=$(tmux list-panes -t $s -F "#{pane_pid}"); echo "$s: $(tr '\0' ' ' < /proc/$P/cmdline | cut -c1-300)"; done
