#!/usr/bin/env bash
# Append two anchored-loss runs at the very end:  ... -> big -> after -> sft2.
# sft2 = C1 (ignore + ordinary) and C7 (mixed security half + ordinary), both with DPO + SFT.
# Only the waiting `after` session is replaced (it no longer removes the hold file; sft2 does).
# To cancel sft2 before it starts:  touch /workspace/process-not-obey/.skip_sft2
set -uo pipefail
W=/workspace/process-not-obey
cd $W
[ -f .big_done ] && { echo "ERROR: big already ended; after may have started. Nothing changed"; exit 1; }
[ -f after.log ] && { echo "ERROR: after already started; nothing changed"; exit 1; }
[ -f .hold_stop ] || { echo "ERROR: hold file missing; nothing changed"; exit 1; }
tmux has-session -t sft2 2>/dev/null && { echo "ERROR: an sft2 session exists already; nothing changed"; exit 1; }
[ -f /workspace/pno-sft/data/generated/seed0/C1_ignore_ordinary.jsonl ] || { echo "ERROR: the sft copy has no C1 data"; exit 1; }
[ -f /workspace/pno-mixsft/data/generated/seed0/C7_mix_ordinary.jsonl ] || { echo "ERROR: the mixsft copy is not prepared"; exit 1; }
tmux kill-session -t after && echo "replaced the waiting after session"
tmux new-session -d -s after "cd $W; while [ ! -f .big_done ]; do sleep 30; done; (bash /workspace/run_after.sh > after.log 2>&1; echo exit=\$? > .after_done)"
tmux new-session -d -s sft2 "cd $W; while [ ! -f .after_done ]; do sleep 30; done; if [ -f .skip_sft2 ]; then echo skipped > .sft2_done; else (bash /workspace/run_cfg_variant.sh sft 'secfid secfid_clean sep conflict' C1_ignore_ordinary > sft2.log 2>&1; r1=\$?; bash /workspace/run_cfg_variant.sh mixsft 'secfid secfid_clean sep conflict' C7_mix_ordinary >> sft2.log 2>&1; echo exit=\$r1,\$? > .sft2_done); fi; rm -f .hold_stop"
sleep 2
tmux ls | cut -c1-40
for s in after sft2; do P=$(tmux list-panes -t $s -F "#{pane_pid}"); echo "$s: $(tr '\0' ' ' < /proc/$P/cmdline | cut -c1-460)"; done
