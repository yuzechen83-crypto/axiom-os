#!/bin/bash
# Axiom-OS x NVIDIA Isaac Sim Launcher (Linux)
#
# Usage: ./run_isaac.sh [options]
# Options:
#   --test          Quick test (100 steps)
#   --steps N       Run for N steps
#   --headless      Run without rendering
#   --no-video      Disable video recording
#   --threshold T   Set discovery threshold

# Configuration - Modify these paths to match your system
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-$HOME/isaac-sim}"
PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Isaac Sim exists
if [ ! -f "$ISAAC_SIM_PATH/python.sh" ]; then
    echo -e "${RED}[ERROR] Isaac Sim not found at $ISAAC_SIM_PATH${NC}"
    echo "Please set ISAAC_SIM_PATH environment variable or edit this script"
    exit 1
fi

# Check if project exists
if [ ! -f "$PROJECT_PATH/run_isaac_demo.py" ]; then
    echo -e "${RED}[ERROR] Project not found at $PROJECT_PATH${NC}"
    exit 1
fi

echo "============================================"
echo "Axiom-OS x NVIDIA Isaac Sim Launcher"
echo "============================================"
echo ""
echo -e "Isaac Sim: ${GREEN}$ISAAC_SIM_PATH${NC}"
echo -e "Project:   ${GREEN}$PROJECT_PATH${NC}"
echo ""

# Collect arguments
ARGS="$@"

echo "Launching with arguments: $ARGS"
echo ""

# Change to project directory
cd "$PROJECT_PATH" || exit 1

# Run with Isaac Sim's Python
"$ISAAC_SIM_PATH/python.sh" "$PROJECT_PATH/run_isaac_demo.py" $ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR] Simulation failed with code $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi

echo ""
echo -e "${GREEN}Simulation completed successfully!${NC}"
