#!/bin/bash

# Tmux script to run CPU experiments with different noise levels and checkpoints
# Updated to include all requested combinations with noise [0,15,30,50] and items [0,20000,40000,60000]

SCRIPT_PATH="/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/code/scripts/cpu_gen.py"
SESSION_NAME="cpu_experiments"
VENV_PATH="/home/jake/Document/Spikes/.linuxvenv"

# Function to create a new tmux session or attach to existing one
create_or_attach_session() {
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Session $SESSION_NAME already exists. Attaching..."
        tmux attach-session -t "$SESSION_NAME"
        exit 0
    else
        echo "Creating new session: $SESSION_NAME"
        tmux new-session -d -s "$SESSION_NAME"
    fi
}

# Create or attach to session
create_or_attach_session

# Define experiments based on your requirements
declare -a experiments=(
    # Format: "noise checkpoint description"
    # Noise 0: Items 500, 1000, 1500, 2500, 5000, 7500, 10000, 20000, 400000, 600000
    "0 500 noise_0_item_500"
    "0 1000 noise_0_item_1000"
    "0 1500 noise_0_item_1500"
    "0 2500 noise_0_item_2500"
    "0 5000 noise_0_item_5000"
    "0 7500 noise_0_item_7500"
    "0 10000 noise_0_item_10000"
    "0 20000 noise_0_item_20000"
)

echo "Setting up tmux panes for experiments..."

# Create first pane (already exists)
tmux rename-window -t "$SESSION_NAME:0" "experiments"

# Run first experiment in the initial pane
first_exp=(${experiments[0]})
noise=${first_exp[0]}
checkpoint=${first_exp[1]}
desc=${first_exp[2]}

tmux send-keys -t "$SESSION_NAME:0.0" "cd /home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/code/scripts" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo 'Starting experiment: $desc'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "source $VENV_PATH/bin/activate && python $SCRIPT_PATH --split test --checkpoint $checkpoint --noise $noise --processes 1" C-m

# Create additional panes for remaining experiments
for i in "${!experiments[@]}"; do
    if [ $i -eq 0 ]; then
        continue  # Skip first experiment (already running)
    fi
    
    exp=(${experiments[$i]})
    noise=${exp[0]}
    checkpoint=${exp[1]}
    desc=${exp[2]}
    
    # Split the window to create a new pane
    tmux split-window -t "$SESSION_NAME:0"
    
    # Run the experiment in the new pane
    tmux send-keys -t "$SESSION_NAME:0.$i" "cd /home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/code/scripts" C-m
    tmux send-keys -t "$SESSION_NAME:0.$i" "echo 'Starting experiment: $desc'" C-m
    tmux send-keys -t "$SESSION_NAME:0.$i" "source $VENV_PATH/bin/activate && python $SCRIPT_PATH --split test --checkpoint $checkpoint --noise $noise --processes 1" C-m
    
    # Arrange panes in a tiled layout
    tmux select-layout -t "$SESSION_NAME:0" tiled
done

echo "All experiments started in tmux session: $SESSION_NAME"
echo "To attach to the session, run: tmux attach-session -t $SESSION_NAME"
echo "To detach from the session, press: Ctrl+b then d"
echo "To kill the session, run: tmux kill-session -t $SESSION_NAME"

# Attach to the session
tmux attach-session -t "$SESSION_NAME"
