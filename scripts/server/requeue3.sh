#!/usr/bin/env bash
# Insert the anchored-loss diagnostic:  seed0 -> nocount -> sft -> extras -> big -> after.
# Only the waiting `extras` session is replaced; nothing that is running is touched.
set -uo pipefail
W=/workspace/process-not-obey
cd $W
[ -f .nocount_done ] && { echo "ERROR: nocount already ended; extras may have started. Nothing changed"; exit 1; }
[ -f extras.log ] && { echo "ERROR: extras already started; nothing changed"; exit 1; }
[ -f .hold_stop ] || { echo "ERROR: hold file missing; nothing changed"; exit 1; }
tmux has-session -t sft 2>/dev/null && { echo "ERROR: an sft session exists already; nothing changed"; exit 1; }
[ -f /workspace/pno-sft/data/generated/seed0/C2_ignore_fidelity.jsonl ] || { echo "ERROR: the sft copy is not prepared"; exit 1; }
tmux kill-session -t extras && echo "replaced the waiting extras session"
tmux new-session -d -s sft "cd $W; while [ ! -f .nocount_done ]; do sleep 30; done; (bash /workspace/run_cfg_variant.sh sft 'secfid secfid_clean conflict' C4_process_fidelity > sft.log 2>&1; r1=\$?; bash /workspace/run_cfg_variant.sh sft 'secfid secfid_clean conflict sep' C2_ignore_fidelity >> sft.log 2>&1; echo exit=\$r1,\$? > .sft_done)"
tmux new-session -d -s extras "cd $W; while [ ! -f .sft_done ]; do sleep 30; done; (bash /workspace/run_extras.sh > extras.log 2>&1; echo exit=\$? > .extras_done)"
sleep 2
tmux ls | cut -c1-40
tmux list-panes -a -F "#{session_name}: #{pane_start_command}" | grep "^sft\|^extras\|^nocount" | cut -c1-330
