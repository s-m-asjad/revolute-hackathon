# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Plays back a trajectory recorded with `lerobot-teleop-record` on a robot.

Unlike `lerobot-replay`, this does not read from a Hugging Face `LeRobotDataset`; it reads
the plain JSON trajectory files produced by `lerobot-teleop-record`.

Example:

```shell
lerobot-playback \
    --robot.type=seeed_b601_rs_follower \
    --robot.port="$PCAN_IF" \
    --robot.id=follower1 \
    --robot.can_adapter=socketcan \
    --robot.joint_directions="{shoulder_pan: 1.0, shoulder_lift: 1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: 1.0, gripper: 3.0}" \
    --trajectory=trajectories/pick_and_place.json
```

If `--trajectory` is omitted, you'll be prompted to choose a file from `--trajectories_dir`
(default `trajectories/`).
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.trajectory_io import list_trajectories, load_trajectory
from lerobot.utils.utils import init_logging, log_say

logger = logging.getLogger(__name__)


@dataclass
class PlaybackConfig:
    robot: RobotConfig
    # Path to a trajectory JSON file saved by `lerobot-teleop-record`. If omitted, you will be
    # prompted to pick one from `trajectories_dir`.
    trajectory: str | None = None
    # Folder to look for trajectory files in when `trajectory` is not given directly.
    trajectories_dir: str = "trajectories"
    # Playback speed multiplier (2.0 = twice as fast, 0.5 = half speed).
    speed: float = 1.0
    # Repeat the trajectory in a loop until interrupted with Ctrl+C.
    loop: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True


def resolve_trajectory_path(cfg: PlaybackConfig) -> Path:
    if cfg.trajectory is not None:
        path = Path(cfg.trajectory)
        if not path.exists():
            candidate = Path(cfg.trajectories_dir) / cfg.trajectory
            if candidate.exists():
                path = candidate
            elif candidate.with_suffix(".json").exists():
                path = candidate.with_suffix(".json")
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {cfg.trajectory}")
        return path

    options = list_trajectories(cfg.trajectories_dir)
    if not options:
        raise FileNotFoundError(
            f"No trajectory files found in '{cfg.trajectories_dir}'. Record one with "
            "`lerobot-teleop-record` first, or pass --trajectory=<path>."
        )

    print(f"\nAvailable trajectories in '{cfg.trajectories_dir}':")
    for i, option in enumerate(options):
        print(f"  [{i}] {option.name}")

    while True:
        choice = input(f"Pick a trajectory to play back [0-{len(options) - 1}]: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(options):
            return options[int(choice)]
        print("Invalid choice, try again.")


def playback_loop(
    robot: Robot,
    frames: list[dict],
    robot_action_processor,
    speed: float,
):
    prev_t = 0.0
    for i, frame in enumerate(frames):
        loop_start = time.perf_counter()

        action = frame["action"]
        obs = robot.get_observation()
        processed_action = robot_action_processor((action, obs))
        robot.send_action(processed_action)

        target_dt = max((frame["t"] - prev_t) / max(speed, 1e-6), 0.0)
        prev_t = frame["t"]

        elapsed = time.perf_counter() - loop_start
        precise_sleep(max(target_dt - elapsed, 0.0))

        print(f"Frame {i + 1}/{len(frames)}", end="\r")
    print()


@parser.wrap()
def playback(cfg: PlaybackConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    path = resolve_trajectory_path(cfg)
    data = load_trajectory(path)
    frames = data["frames"]

    if not frames:
        logging.warning(f"'{path}' has no frames, nothing to play back.")
        return

    if data.get("robot_type") and data["robot_type"] != cfg.robot.type:
        logging.warning(
            f"Trajectory was recorded on robot type '{data['robot_type']}' but you're playing it "
            f"back on '{cfg.robot.type}'. Continuing anyway."
        )

    robot_action_processor = make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    try:
        log_say(f"Playing back {path.name}", cfg.play_sounds)
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Playback pass {iteration} ({len(frames)} frames, {data['duration_s']:.2f}s) ---")
            playback_loop(robot, frames, robot_action_processor, cfg.speed)
            if not cfg.loop:
                break
        log_say("Done playing back trajectory", cfg.play_sounds)
    except KeyboardInterrupt:
        log_say("Playback interrupted", cfg.play_sounds)
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    playback()


if __name__ == "__main__":
    main()
