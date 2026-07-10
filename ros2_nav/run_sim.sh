#!/usr/bin/env bash
# Launch the full ROS 2 nav loop against MuJoCo: bridge (sensors+motors) + policy node.
# Usage: ros2_nav/run_sim.sh [seconds]   (default 60; 0 = run until Ctrl-C)
cd "$(dirname "$0")/.."
source /opt/ros/jazzy/setup.bash
source ros2_venv/bin/activate
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
SECS="${1:-60}"
BLOG=/tmp/ros2_bridge.log; PLOG=/tmp/ros2_policy.log

python ros2_nav/mujoco_bridge.py > "$BLOG" 2>&1 &
BPID=$!
python ros2_nav/policy_node.py > "$PLOG" 2>&1 &
POLPID=$!
echo "bridge PID $BPID | policy PID $POLPID | logs: $BLOG $PLOG"

if [ "$SECS" = "0" ]; then
  wait
else
  sleep "$SECS"
  kill $BPID $POLPID 2>/dev/null
  echo "=== stopped after ${SECS}s ==="
  R=$(grep -c REACHED "$BLOG"); T=$(grep -c timeout "$BLOG")
  echo "REACHED: $R | timeouts: $T | success = $(python3 -c "print(f'{$R/max($R+$T,1):.0%}')")"
  echo "--- policy node startup ---"; grep -iE "loaded|goal|error|traceback" "$PLOG" | head
  echo "--- bridge tail ---"; grep -iE "REACHED|timeout|error|traceback" "$BLOG" | tail -5
fi
