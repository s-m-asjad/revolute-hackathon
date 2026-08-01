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
Teleoperate a robot while recording named trajectories to simple JSON files on disk.

This is a lightweight alternative to `lerobot-record`: it doesn't require a Hugging Face
dataset repo, cameras, or task descriptions. It just continuously teleoperates the robot
(exactly like `lerobot-teleoperate`) and lets you tap keys to mark the start/end of a
trajectory. Each saved trajectory is a plain JSON file containing the sequence of actions
that were sent to the follower, which can later be replayed with `lerobot-playback`.

Example:

```shell
lerobot-teleop-record \
    --robot.type=seeed_b601_rs_follower \
    --robot.port="$PCAN_IF" \
    --robot.id=follower1 \
    --robot.can_adapter=socketcan \
    --robot.joint_directions="{shoulder_pan: 1.0, shoulder_lift: 1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: 1.0, gripper: 3.0}" \
    --teleop.type=rebot_arm_102_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_arm_102_leader
```

Controls (needs pynput to be able to capture keyboard events from the terminal):
    r        - start recording a new trajectory
    s        - stop recording the current trajectory (you'll then be asked for a filename)
    m        - toggle "manual joint" mode: while enabled, the follower ignores the leader
               arm entirely and only moves the joint set by --manual_joint (default
               `wrist_roll`, i.e. joint id 6 on the seeed_b601 arms -- the joint the
               gripper is directly mounted on). Press again to hand control back to the
               leader arm.
    Up / Down - while in manual joint mode, increase / decrease that joint's angle by
               --manual_jog_step_deg degrees per press
    q / Esc  - quit

Trajectories are saved as `<output_dir>/<name>.json` (default `output_dir` is `trajectories/`
in the current directory). Play them back with:

```shell
lerobot-playback --robot.type=... --robot.port=... --trajectory=trajectories/<name>.json
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
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
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.control_utils import is_headless
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.trajectory_io import save_trajectory
from lerobot.utils.utils import init_logging, log_say, move_cursor_up

logger = logging.getLogger(__name__)


@dataclass
class TeleopRecordConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    # Folder where trajectory JSON files get saved.
    output_dir: str = "trajectories"
    # Joint controlled by Up/Down while in manual joint mode ('m' key). Defaults to
    # `wrist_roll`, which is joint id 6 on the seeed_b601 arms (CAN send id 0x06) -- the
    # joint the gripper is directly mounted on.
    manual_joint: str = "wrist_roll"
    # Degrees added/subtracted from the manual joint's real-world angle per Up/Down keypress.
    manual_jog_step_deg: float = 2.0
    # Use vocal synthesis to read events.
    play_sounds: bool = True


class TrajectoryRecorder:
    """Tracks recording state and buffers frames while teleoperating."""

    def __init__(self, joint_names: list[str]):
        self.joint_names = joint_names
        self.is_recording = False
        self._frames: list[dict] = []
        self._start_t: float = 0.0

    def start(self) -> None:
        self._frames = []
        self._start_t = time.perf_counter()
        self.is_recording = True

    def add_frame(self, action: RobotAction) -> None:
        if not self.is_recording:
            return
        t = time.perf_counter() - self._start_t
        self._frames.append({"t": t, "action": dict(action)})

    def stop(self) -> list[dict]:
        self.is_recording = False
        frames = self._frames
        self._frames = []
        return frames


def init_record_keyboard_listener():
    """
    Sets up a non-blocking keyboard listener with the controls documented at the top of this file.

    Returns a tuple of (listener, events) where events is a dict of flags that the main loop should
    poll and reset after handling.
    """
    events = {
        "start_recording": False,
        "stop_recording": False,
        "toggle_manual": False,
        "jog_increase": 0,
        "jog_decrease": 0,
        "quit": False,
    }

    if is_headless():
        logging.warning(
            "Headless environment detected. Keyboard controls will not be available; "
            "this script has no way to start/stop recording without a keyboard listener."
        )
        return None, events

    from pynput import keyboard as pynput_keyboard

    def on_press(key):
        try:
            if hasattr(key, "char") and key.char == "r":
                events["start_recording"] = True
            elif hasattr(key, "char") and key.char == "s":
                events["stop_recording"] = True
            elif hasattr(key, "char") and key.char == "m":
                events["toggle_manual"] = True
            elif hasattr(key, "char") and key.char == "q":
                events["quit"] = True
            elif key == pynput_keyboard.Key.up:
                events["jog_increase"] += 1
            elif key == pynput_keyboard.Key.down:
                events["jog_decrease"] += 1
            elif key == pynput_keyboard.Key.esc:
                events["quit"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def reset_transient_events(events: dict) -> None:
    """Clears flags that may have been set by stray keypresses while a blocking input() was up."""
    events["start_recording"] = False
    events["stop_recording"] = False
    events["toggle_manual"] = False
    events["jog_increase"] = 0
    events["jog_decrease"] = 0


def prompt_for_filename(output_dir: Path) -> str | None:
    """Blocks on stdin asking for a filename to save under. Returns None if the user wants to discard."""
    while True:
        name = input(
            "\nEnter a name to save this trajectory (without extension), "
            "or leave blank to discard it: "
        ).strip()
        if not name:
            return None
        filename = name if name.endswith(".json") else f"{name}.json"
        dest = output_dir / filename
        if dest.exists():
            overwrite = input(f"'{dest}' already exists. Overwrite? [y/N]: ").strip().lower()
            if overwrite != "y":
                continue
        return filename


def get_joint_direction(robot: Robot, joint_name: str) -> float:
    """Looks up the follower's configured sign/scale for a joint, defaulting to 1.0 if unknown."""
    direction = getattr(robot.config, "joint_directions", {}).get(joint_name)
    if not direction:
        return 1.0
    return direction


def action_from_observation(robot: Robot, obs: RobotObservation) -> RobotAction:
    """Builds an action dict (in the same pre-transform space `robot.send_action` expects) that
    holds every joint at its current real-world position, e.g. to freeze the arm in place."""
    action: RobotAction = {}
    for key in robot.action_features:
        joint_name = key.removesuffix(".pos")
        real_pos = obs.get(key, 0.0)
        action[key] = real_pos / get_joint_direction(robot, joint_name)
    return action


def teleop_record_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    output_dir: Path,
    manual_joint: str,
    manual_jog_step_deg: float,
    play_sounds: bool,
):
    recorder = TrajectoryRecorder(joint_names=[k.removesuffix(".pos") for k in robot.action_features])
    listener, events = init_record_keyboard_listener()

    manual_key = f"{manual_joint}.pos"
    manual_direction = get_joint_direction(robot, manual_joint)
    manual_mode = False
    manual_target: float = 0.0
    # Snapshot of every joint's position (in pre-transform action space) taken the moment manual
    # mode is enabled; reused each frame with only `manual_joint` overridden, so the rest of the
    # arm stays put while the leader is ignored.
    frozen_action: RobotAction = {}

    display_len = max(len(key) for key in robot.action_features)
    help_text = (
        "Press 'r' to start recording, 's' to stop and save, "
        "'m' to toggle manual control of "
        f"'{manual_joint}' (Up/Down to jog), 'q' to quit."
    )
    log_say("Ready. " + help_text, play_sounds)
    print("\n" + help_text + "\n")

    try:
        while not events["quit"]:
            loop_start = time.perf_counter()

            if events["start_recording"]:
                events["start_recording"] = False
                if not recorder.is_recording:
                    recorder.start()
                    log_say("Recording started", play_sounds)
                    print("\n[recording started]")

            if events["stop_recording"]:
                events["stop_recording"] = False
                if recorder.is_recording:
                    frames = recorder.stop()
                    log_say("Recording stopped", play_sounds)
                    duration = frames[-1]["t"] if frames else 0.0
                    print(f"\n[recording stopped] {len(frames)} frames, {duration:.2f}s")
                    filename = prompt_for_filename(output_dir)
                    if filename is None:
                        print("Discarded.")
                    else:
                        dest = output_dir / filename
                        save_trajectory(
                            dest,
                            joint_names=recorder.joint_names,
                            fps=fps,
                            robot_type=robot.robot_type,
                            teleop_type=teleop.name,
                            frames=frames,
                        )
                        print(f"Saved to {dest}")
                    # Typing the filename may have brushed keys we react to (r/s/m); ignore those.
                    reset_transient_events(events)
                    print("\n" + help_text + "\n")

            if events["toggle_manual"]:
                events["toggle_manual"] = False
                manual_mode = not manual_mode
                if manual_mode:
                    # Freeze every joint at its current real-world position, and seed the jog
                    # target so the first Up/Down press moves it relative to where it already is.
                    current_obs = robot.get_observation()
                    frozen_action = action_from_observation(robot, current_obs)
                    manual_target = frozen_action[manual_key]
                    log_say(f"Manual control of {manual_joint} enabled", play_sounds)
                    print(f"\n[MANUAL MODE] Leader ignored. Jogging '{manual_joint}' with Up/Down.")
                else:
                    log_say("Leader control resumed", play_sounds)
                    print(f"\n[LEADER MODE] '{manual_joint}' and all other joints follow the leader again.")

            # Get robot observation (needed by the processor pipelines' signature)
            obs = robot.get_observation()

            if manual_mode:
                jog_steps = events["jog_increase"] - events["jog_decrease"]
                events["jog_increase"] = 0
                events["jog_decrease"] = 0
                if jog_steps:
                    manual_target += (jog_steps * manual_jog_step_deg) / manual_direction

                teleop_action = dict(frozen_action)
                teleop_action[manual_key] = manual_target
            else:
                # Get teleop action
                raw_action = teleop.get_action()
                # Process teleop action through pipeline
                teleop_action = teleop_action_processor((raw_action, obs))

            # Process action for robot through pipeline
            robot_action_to_send = robot_action_processor((teleop_action, obs))

            # Send processed action to robot
            _ = robot.send_action(robot_action_to_send)

            # Record the pre-robot-processor action, mirroring what `lerobot-record` stores,
            # so that `lerobot-playback` can push it back through the same robot_action_processor.
            recorder.add_frame(teleop_action)

            rec_status = "RECORDING" if recorder.is_recording else "idle"
            ctrl_status = f"MANUAL:{manual_joint}" if manual_mode else "LEADER"
            print(f"\n{rec_status:<10} | {ctrl_status:<20} | {'-' * display_len}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 2)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))
    finally:
        if listener is not None:
            listener.stop()


@parser.wrap()
def teleop_record(cfg: TeleopRecordConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, _ = make_default_processors()

    valid_joints = [k.removesuffix(".pos") for k in robot.action_features]
    if cfg.manual_joint not in valid_joints:
        raise ValueError(
            f"--manual_joint={cfg.manual_joint!r} is not a joint of this robot. "
            f"Valid options are: {valid_joints}"
        )

    teleop.connect()
    robot.connect()

    try:
        teleop_record_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            output_dir=output_dir,
            manual_joint=cfg.manual_joint,
            manual_jog_step_deg=cfg.manual_jog_step_deg,
            play_sounds=cfg.play_sounds,
        )
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleop_record()


if __name__ == "__main__":
    main()
