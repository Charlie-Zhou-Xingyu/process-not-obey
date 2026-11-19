#!/usr/bin/env bash
# Insert the no-count diagnostic into the chain:  seed0 -> nocount -> extras -> big -> after.
# Only the waiting `extras` session is replaced; nothing that is running is touched.
set -uo pipefail
W=/workspace/process-not-obey
cd $W
[ -f .seed0_done ] && { echo "ERROR: seed0 already ended ($(cat .seed0_done)); nothing changed"; exit 1; }
[ -f extras.log ] && { echo "ERROR: extras already started; nothing changed"; exit 1; }
[ -f .hold_stop ] || { echo "ERROR: hold file missing; nothing changed"; exit 1; }
tmux has-session -t nocount 2>/dev/null && { echo "ERROR: a nocount session exists already; nothing changed"; exit 1; }
tmux kill-session -t extras && echo "replaced the waiting extras session"
tmux new-session -d -s nocount "cd $W; while [ ! -f .seed0_done ]; do sleep 30; done; (bash /workspace/run_keep_variant.sh nocount '^count_' C3_process_ordinary C4_process_fidelity > nocount.log 2>&1; echo exit=\$? > .nocount_done)"
tmux new-session -d -s extras "cd $W; while [ ! -f .nocount_done ]; do sleep 30; done; (bash /workspace/run_extras.sh > extras.log 2>&1; echo exit=\$? > .extras_done)"
sleep 2
tmux ls
tmux list-panes -a -F "#{session_name}: #{pane_start_command}" | grep -v "^main\|^ssh_tmux" | cut -c1-250
