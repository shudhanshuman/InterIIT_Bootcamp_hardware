# Clue Chain Hunt — Solver (`clue_hunt_solver`)

Autonomous ROS 2 Humble solution for the Clue Chain Hunt bootcamp:
a **leader** buggy reads a chained sequence of ArUco+QR boards
(`HUNT:<id>:<token>:<command>`, token = `SHA1(prev)[:4]`), navigates the chain
with Nav2, ignores decoy / look-alike boards, and drives onto the treasure —
while a camera-only **follower** buggy trails the leader's back tag (ArUco 49).

Stack: Ubuntu 22.04 · ROS 2 Humble · Gazebo Fortress 6.x · Nav2 · slam_toolbox · OpenCV.
Workspace: `~/hunt_ws`. Design authority: `final_plan.md`. Command runbook: `COMMANDS.md`.
Phases: `phase_1.md` (map+Nav2), `phase_2.md` (vision), `phase_3.md` (leader chain),
`last_phase.md` (follower).

## 1. Packages and nodes

| Package | Role |
|---|---|
| `clue_hunt_description` | URDF/xacro for both robots + tag mesh (do not modify) |
| `clue_hunt_gazebo` | Practice world `practice.sdf`, board models, sim launch (do not modify) |
| `clue_hunt_navigation` | slam_toolbox + Nav2 configs and launch files |
| `clue_hunt_solver` | **Our solution** — `hunt_node`, `follower_node`, `hunt.launch.py`, cleaned map, Nav2 params copy |

Key topics: `/hunt/clues` (`std_msgs/String`, valid clue texts in chain order),
`/hunt/boards` (`"<id> <x> <y>"` in `map` frame), `/hunt/treasure`
(`geometry_msgs/PoseStamped`, frame `map`, published once), `/leader/status`
(`MOVING`/`SEARCHING`/`READING`/`DONE`), `/follower/cmd_vel`.

Rules the code obeys: leader never uses Gazebo ground truth or hard-coded
board/pillar/treasure positions; follower uses only `/follower/camera/*`,
`/follower/odom`, `follower/*` TF and `/leader/status`; Nav2 owns `/cmd_vel`.

## 2. Prerequisites and install

```bash
# ROS 2 Humble + Gazebo Fortress + Nav2/SLAM/vision (one time)
sudo apt update
sudo apt install -y \
  ros-humble-ros-gz ros-humble-xacro ros-humble-robot-state-publisher \
  ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-nav2-simple-commander \
  ros-humble-slam-toolbox ros-humble-teleop-twist-keyboard \
  ros-humble-cv-bridge ros-humble-rqt-image-view ros-humble-tf2-tools \
  python3-opencv python3-numpy ignition-fortress libzbar0

# Python deps of our solver package
cd ~/hunt_ws
pip install -r requirements.txt
```

## 3. Build (every terminal needs the sources)

```bash
cd ~/hunt_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select clue_hunt_solver
source install/setup.bash
```

Optional convenience (append once to `~/.bashrc`):

```bash
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'source ~/hunt_ws/install/setup.bash' >> ~/.bashrc
```

Check installed files ship inside our package (evaluators replace the other packages):

```bash
SHARE=$(ros2 pkg prefix --share clue_hunt_solver)
ls $SHARE/maps/arena.yaml $SHARE/config/nav2_params.yaml \
   $SHARE/config/search_viewpoints.yaml $SHARE/launch/hunt.launch.py
```

## 4. How to run — video demo (copy-paste in order)

Three terminals. **Every terminal starts with the same clean environment**
(DDS discovery splits if terminals differ — this is the `~/.bashrc` leftover
issue from `last_phase.md`: stale `ROS_DOMAIN_ID`, Cyclone/FastDDS mix, old IPs):

```bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=0
unset RMW_IMPLEMENTATION
source /opt/ros/humble/setup.bash && source ~/hunt_ws/install/setup.bash
```

### Step 0 — kill any stale stack (ALWAYS first; stale processes are the #1 false alarm)

```bash
pkill -f "hunt_node|follower_node|hunt.launch|navigation.launch|mapping.launch|sim.launch|gazebo|gz sim|rviz2|ros_gz_bridge|parameter_bridge|nav2_container" ; sleep 3
ps aux | grep -E "gazebo|gz sim|hunt_node|bt_navigator|nav2_container|rviz2" | grep -v grep || echo "bus clear"
ros2 daemon stop
ros2 topic info /clock  # must show Publisher count: 1 (2 = duplicate stack, kill again)
```

### Step 1 — T1: full stack, one command (sim + AMCL + Nav2 + both nodes, headless)

```bash
ros2 launch clue_hunt_solver hunt.launch.py start_nav:=true
```

### Step 2 — T2: watch the clues (same clean env first, start AFTER T1 shows `hunt_node ready`)

```bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=0
unset RMW_IMPLEMENTATION
source /opt/ros/humble/setup.bash && source ~/hunt_ws/install/setup.bash
ros2 topic echo /hunt/clues
```

### Step 3 — T3: watch the treasure (same clean env first, message arrives once at the end)

```bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=0
unset RMW_IMPLEMENTATION
source /opt/ros/humble/setup.bash && source ~/hunt_ws/install/setup.bash
ros2 topic echo /hunt/treasure
```

(Your note pasted `ros2 topic echo /hunt/clues` for both T2 and T3 — T2 is the
clues stream, T3 is the treasure pose; this matches `phase_3.md` Step 4 and
`last_phase.md` §8. Leave both echoes open for the whole run.)

Then **hands off** — no teleop, no RViz clicks. Expect ~3–10 minutes total.

### Step 4 — what success looks like

`/hunt/clues` prints exactly these five lines, in order, nothing else
(neither `HUNT:7:...` decoy nor `HUNT:4:07A8:...` look-alike may appear):

```
HUNT:1:7196:GOTO 1.5 -3.9
HUNT:2:7F49:PILLAR RED
HUNT:3:BCB8:BETWEEN BLUE GREEN 0.59
HUNT:4:EC9E:REL 5.08 -0.65
HUNT:5:1756:TREASURE REL 0.98 3.15
```

T3 treasure pose must be within 0.3 m of the true treasure (aim <= 0.15 m),
`/leader/status` ends on `DONE`, follower stays 0.6–2.0 m and ends within
1.5 m of treasure. Per-run CSV logs (`/tmp/hunt_run_*.csv`,
`/tmp/follower_run_*.csv`, paths printed at node startup) hold the
state machine, every detection/verdict, colour assignment gap (>= 0.15),
and distance trace for the report.

## 5. Appendix — all other commands you may need

```bash
# --- setup, every terminal ---
source /opt/ros/humble/setup.bash && source ~/hunt_ws/install/setup.bash

# --- build ---
cd ~/hunt_ws && colcon build --symlink-install
colcon build --symlink-install --packages-select clue_hunt_solver

# --- unit tests (94 tests: clue/frames/vision/outlines/search/pillars/stack) ---
cd ~/hunt_ws && source install/setup.bash
python3 -m unittest discover -s src/clue_hunt_solver/test -v 2>&1 | tail -5
# expect: Ran 94 tests ... OK
# fallback if install not sourced:
PYTHONPATH=src/clue_hunt_solver python3 -m unittest discover -s src/clue_hunt_solver/test -v

# --- sim only variants ---
ros2 launch clue_hunt_gazebo sim.launch.py
ros2 launch clue_hunt_gazebo sim.launch.py follower:=false   # leader only (mapping)
ros2 launch clue_hunt_gazebo sim.launch.py gui:=false        # headless
ros2 launch clue_hunt_gazebo sim.launch.py world:=<name> world_pkg:=<pkg>

# --- mapping (once; raw map -> ~/hunt_ws/maps/arena.{yaml,pgm}) ---
ros2 launch clue_hunt_navigation mapping.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard        # drive slowly
ros2 run nav2_map_server map_saver_cli -f ~/hunt_ws/maps/arena

# --- map cleanup (Phase 1; cleaned map lands in src/clue_hunt_solver/maps/) ---
python3 src/clue_hunt_solver/scripts/clean_map.py
python3 src/clue_hunt_solver/scripts/verify_map.py \
  --expected-pillars "5.0,-2.0 5.5,2.5 2.2,2.8" --practice   # expect ALL CHECKS PASSED

# --- navigation WITHOUT our nodes (evaluator style) ---
SHARE=$(ros2 pkg prefix --share clue_hunt_solver)
ros2 launch clue_hunt_navigation navigation.launch.py \
  map:=$SHARE/maps/arena.yaml params_file:=$SHARE/config/nav2_params.yaml
# if sim already running elsewhere, add: sim:=false

# --- solution WITHOUT Nav2 bringup (Nav2 already up) ---
ros2 launch clue_hunt_solver hunt.launch.py
ros2 run clue_hunt_solver hunt_node                         # leader alone
ros2 run clue_hunt_solver follower_node                     # follower alone

# --- L2 tour rebuild (Phase 3) ---
python3 src/clue_hunt_solver/scripts/build_tour.py
colcon build --symlink-install --packages-select clue_hunt_solver && source install/setup.bash
grep -c x: install/clue_hunt_solver/share/clue_hunt_solver/config/search_viewpoints.yaml  # 10

# --- send a Nav2 goal by hand (no RViz click) ---
ros2 action send_goal -f /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 2.8, y: 1.5}, orientation: {w: 1.0}}}}"

# --- debug / inspection ---
ros2 topic list && ros2 node list
ros2 topic echo /hunt/clues std_msgs/msg/String            # typed form (beats discovery race)
ros2 topic echo /hunt/treasure geometry_msgs/msg/PoseStamped
ros2 topic echo /leader/status
ros2 topic hz /camera/image_raw
ros2 topic info /cmd_vel                                    # Nav2 only; silent while idle
ros2 lifecycle get /bt_navigator
ros2 run tf2_ros tf2_echo map base_footprint
ros2 run tf2_tools view_frames                              # -> frames.pdf
ros2 run rqt_image_view rqt_image_view                      # /camera/image_raw
ros2 run rqt_image_view rqt_image_view /follower/debug/image  # follower overlay (green tag box)
ros2 topic echo /follower/cmd_vel                           # 20 Hz Twist while following
ros2 run teleop_twist_keyboard teleop_twist_keyboard
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/follower/cmd_vel

# --- follower YOLO verify (healthy machine only, weights required; torch core-dumps in some VMs) ---
ros2 run clue_hunt_solver follower_node --ros-args -p use_sim_time:=true \
  -p use_yolo:=true -p yolo_weights:=/home/robo/hunt_ws/src/clue_hunt_solver/models/leader_yolo.pt

# --- treasure error check (paste x,y from /hunt/treasure echo) ---
python3 - <<'EOF'
import math
p = (8.52, -3.65)       # <- paste actual x,y
truth = (8.5, -3.5)     # practice.sdf pose (report-side check only)
print(f'treasure error: {math.hypot(p[0]-truth[0], p[1]-truth[1]):.3f} m  (limit 0.3, aim <= 0.15)')
EOF
```

Troubleshooting quick table: `Package not found` → source the install setup;
black camera / Gazebo crash in VM → `export LIBGL_ALWAYS_SOFTWARE=1`, `gui:=false`;
Nav2 waiting → it auto-poses at (0,0,0), never click 2D Pose Estimate in eval runs;
`xmlrpc Fault !rclpy.ok()` on echo → fresh shell, `ros2 daemon stop`, retry;
two `/clock` publishers / `jump back in time` → kill-all (Step 0), launch exactly once.
Full tables: `COMMANDS.md` §11, `phase_3.md` §8, `last_phase.md` §7.
