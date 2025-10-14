#!/usr/bin/env bash
# launch_tmux.sh
# Creates a tmux session "mnist_run" with one window per batch set
# and two panes per window, each running a cpu_gen.py process using absolute paths.

# Absolute path to the script directory
SCRIPT_DIR="$HOME/Document/Spikes/projects/mnist_class/mnist_class_wip/code/scripts"

SESSION="mnist_run"

# Start a new, detached tmux session with the first window
# Name format: c<CHECKPOINT>_n<NOISE>_<START>-<END>
tmux new-session -d -s $SESSION -n "c0_n0_742-1000"

# Helper function to set up a window
setup_set() {
  local idx=$1 name=$2 cmd0=$3 cmd1=$4
  tmux new-window -t $SESSION:$idx -n "$name"
  tmux split-window -t $SESSION:$idx -v
  tmux select-window -t $SESSION:$idx
  tmux select-pane -t 0
  tmux send-keys "cd $SCRIPT_DIR && $cmd0" C-m
  tmux select-pane -t 1
  tmux send-keys "cd $SCRIPT_DIR && $cmd1" C-m
}

# Configure each batch set
setup_set 0 "c0_n0_742-1000" \
  "python cpu_gen.py --split train --checkpoint 0 --noise 0 --start_batch 742 --end_batch 870 --processes 1" \
  "python cpu_gen.py --split train --checkpoint 0 --noise 0 --start_batch 871 --end_batch 1000 --processes 1"

setup_set 1 "c0_n5_765-1200" \
  "python cpu_gen.py --split train --checkpoint 0 --noise 5 --start_batch 765 --end_batch 981 --processes 1" \
  "python cpu_gen.py --split train --checkpoint 0 --noise 5 --start_batch 982 --end_batch 1200 --processes 1"

setup_set 2 "c60000_n5_241-600" \
  "python cpu_gen.py --split train --checkpoint 60000 --noise 5 --start_batch 241 --end_batch 420 --processes 1" \
  "python cpu_gen.py --split train --checkpoint 60000 --noise 5 --start_batch 421 --end_batch 600 --processes 1"

setup_set 3 "c60000_n5_836-1200" \
  "python cpu_gen.py --split train --checkpoint 60000 --noise 5 --start_batch 836 --end_batch 1017 --processes 1" \
  "python cpu_gen.py --split train --checkpoint 60000 --noise 5 --start_batch 1018 --end_batch 1200 --processes 1"

setup_set 4 "c60000_n0_237-500" \
  "python cpu_gen.py --split train --checkpoint 60000 --noise 0 --start_batch 237 --end_batch 368 --processes 1" \
  "python cpu_gen.py --split train --checkpoint 60000 --noise 0 --start_batch 369 --end_batch 500 --processes 1"

setup_set 5 "c60000_n0_735-1200" \
  "python cpu_gen.py --split train --checkpoint 60000 --noise 0 --start_batch 735 --end_batch 967 --processes 1" \
  "python cpu_gen.py --split train --checkpoint 60000 --noise 0 --start_batch 968 --end_batch 1200 --processes 1"

# Attach to the session
tmux attach -t $SESSION
