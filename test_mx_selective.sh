#!/bin/bash

# Test script for mx_lm_selective.py
# This script starts a tmux session and runs tests with GPU monitoring

SESSION_NAME="mx_selective_test"

# Check if tmux session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    echo "Creating new tmux session: $SESSION_NAME"
    
    # Create new session with first window for the main script
    tmux new-session -d -s $SESSION_NAME -n "mx_test"
    
    # Create second window for GPU monitoring
    tmux new-window -t $SESSION_NAME -n "gpu_monitor"
    
    # In the monitoring window, run watch nvidia-smi
    tmux send-keys -t $SESSION_NAME:gpu_monitor "watch -n 2 nvidia-smi" C-m
    
    # Go back to the first window
    tmux select-window -t $SESSION_NAME:mx_test
    
    echo "Tmux session created. Windows:"
    echo "  0: mx_test (main script)"
    echo "  1: gpu_monitor (nvidia-smi watch)"
    echo ""
    echo "To attach: tmux attach -t $SESSION_NAME"
    echo "To detach: Ctrl-b d"
    echo "To switch windows: Ctrl-b 0 (or 1)"
else
    echo "Tmux session '$SESSION_NAME' already exists"
    echo "To attach: tmux attach -t $SESSION_NAME"
fi

echo ""
echo "Starting test in tmux session..."

# Navigate to the project directory
tmux send-keys -t $SESSION_NAME:mx_test "cd /scratch/ashvin/mx-scaling-tests" C-m

# Run tests based on argument
if [ "$1" == "quick" ]; then
    echo "Running quick test with tinystories-33M..."
    tmux send-keys -t $SESSION_NAME:mx_test "echo '=== Quick Test: tinystories-33M (MLP only) ==='" C-m
    tmux send-keys -t $SESSION_NAME:mx_test "uv run python mx_lm_selective.py --models tinystories-33M --mlp_only --max_samples 100 --batch_size 4 --min_free_ratio 0.3" C-m
elif [ "$1" == "medium" ]; then
    echo "Running medium test with 7B model..."
    tmux send-keys -t $SESSION_NAME:mx_test "echo '=== Medium Test: qwen2.5-7b (MLP only) ==='" C-m
    tmux send-keys -t $SESSION_NAME:mx_test "uv run python mx_lm_selective.py --models qwen2.5-7b --mlp_only --max_samples 500 --batch_size 4 --min_free_ratio 0.3" C-m
elif [ "$1" == "full" ]; then
    echo "Running full test with multiple models..."
    tmux send-keys -t $SESSION_NAME:mx_test "echo '=== Full Test: All Small Models ==='" C-m
    tmux send-keys -t $SESSION_NAME:mx_test "uv run python mx_lm_selective.py --models tinystories-33M qwen1.5-0.5B phi-1_5 --mlp_only --max_samples 1000 --min_free_ratio 0.3" C-m
elif [ "$1" == "large" ]; then
    echo "Running large model test..."
    tmux send-keys -t $SESSION_NAME:mx_test "echo '=== Large Test: 7B-14B Models ==='" C-m
    tmux send-keys -t $SESSION_NAME:mx_test "uv run python mx_lm_selective.py --models qwen2.5-7b qwen2.5-14b llama-3.1-8b --mlp_only --max_samples 1000 --min_free_ratio 0.2" C-m
else
    echo "Usage: $0 [quick|medium|full|large]"
    echo "  quick  - Fast test with tiny model (tinystories-33M)"
    echo "  medium - Test with 7B model (qwen2.5-7b)"
    echo "  full   - Test all small models"
    echo "  large  - Test 7B-14B models"
    echo ""
    echo "Running default quick test..."
    tmux send-keys -t $SESSION_NAME:mx_test "echo '=== Default Quick Test ==='" C-m
    tmux send-keys -t $SESSION_NAME:mx_test "uv run python mx_lm_selective.py --models tinystories-33M --mlp_only --max_samples 100 --batch_size 4 --min_free_ratio 0.3" C-m
fi

echo ""
echo "Test started! Monitor progress with:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "Or check the log file:"
echo "  tail -f mx_selective_log.txt"

