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
TEST FEATURE: uses a local LLM (via Ollama) to decide whether a free-text prompt is asking for one
specific task. If it is, and a camera is configured, it first takes a photo and asks a local
vision-language model (Qwen2.5-VL, also via Ollama) what it sees and whether the task's required
ingredients/items are actually on the table -- this is printed to the console either way. Every
trajectory file in `--trajectories_dir` then gets played back in sequence. If the prompt doesn't match,
nothing happens -- the robot is never touched.

By default the ingredient check is report-only (`--require_ingredients=false`): it always prints what it
sees and which ingredients are present/missing, then proceeds to play back the trajectories regardless.
Pass `--require_ingredients=true` once you want it to actually refuse to run when something's missing.

This is intentionally simple (single hardcoded task, no per-trajectory catalog/selection yet) so the
local-LLM plumbing can be validated on real hardware before building the full "pick which trajectory(ies)
match" version.

Example:

```shell
lerobot-play-by-prompt \
    --robot.type=seeed_b601_rs_follower \
    --robot.port="$PCAN_IF" \
    --robot.id=follower1 \
    --robot.can_adapter=socketcan \
    --robot.joint_directions="{shoulder_pan: 1.0, shoulder_lift: 1.0, elbow_flex: -1.0, wrist_flex: -1.0, wrist_yaw: -1.0, wrist_roll: 1.0, gripper: 3.0}" \
    --prompt="make a sandwich" \
    --camera=/dev/video2 \
    --ingredients="[bread, cheese, ham]"
```

If Ollama isn't already running, this starts it automatically (`ollama serve`) and pulls whichever
model(s) are needed the first time they're used (default text model `qwen2.5:3b`, ~2GB; default vision
model `qwen2.5vl:3b`, ~3.2GB). Pass `--camera=""` to skip the photo/ingredient check entirely.

Instead of typing `--prompt`, pass `--voice=true` to speak the instruction into a mic instead: it records
until you press Enter, then transcribes locally with Whisper (`pip install lerobot[voice]`) -- no cloud
service, no API key. Say the wake phrase "Hey Chef" (configurable via `--voice_wake_word`) before your
instruction, e.g. "Hey Chef, make a sandwich" -- only the part after the wake phrase is used as the
prompt. If the wake phrase isn't heard, the recording is discarded (nothing runs). The extracted
instruction is then used exactly like `--prompt` would be:

```shell
lerobot-play-by-prompt \
    --robot.type=seeed_b601_rs_follower \
    --robot.port="$PCAN_IF" \
    --robot.id=follower1 \
    --robot.can_adapter=socketcan \
    --voice=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing
from PIL import Image as PILImage

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
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
from lerobot.scripts.lerobot_playback import playback_loop
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.ollama_client import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_VLM_MODEL,
    analyze_table,
    classify_match,
    ensure_model_available,
    ensure_server_running,
    unload_model,
)
from lerobot.utils.trajectory_io import list_trajectories, load_trajectory
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.voice_input import extract_after_wake_word, record_and_transcribe

logger = logging.getLogger(__name__)


@dataclass
class PlayByPromptConfig:
    robot: RobotConfig
    # Free-text instruction to classify, e.g. "make a sandwich". Required unless `voice=true`, in which
    # case the instruction is captured from the mic instead.
    prompt: str | None = None
    # Instead of reading `prompt` from the command line, record from the mic and transcribe it locally
    # with Whisper (`pip install lerobot[voice]`). The transcript is then used exactly like `prompt`.
    voice: bool = False
    # Safety cap (in seconds) on mic recording when `voice=true` -- recording actually stops as soon as
    # you press Enter, this only kicks in if you never do.
    voice_max_duration: float = 60.0
    # Local Whisper model size used for transcription (only relevant when `voice=true`).
    voice_model: str = "tiny.en"
    # Speech-to-text backend: "auto" (prefers faster-whisper, falls back to openai-whisper), "faster", or
    # "openai".
    voice_engine: str = "auto"
    # Language code for Whisper transcription.
    voice_language: str = "en"
    # sounddevice input device index to record from. Run with `--voice_list_devices=true` to see options.
    voice_device: int | None = None
    # Print available microphones and exit (does not connect to the robot).
    voice_list_devices: bool = False
    # Wake phrase the actual instruction must follow, e.g. "make me a sandwich" in "Hey Chef, make me a
    # sandwich" only counts starting after "Hey Chef". Case-insensitive. If the phrase isn't heard at all,
    # the recording is discarded (treated as not directed at the robot) instead of falling back to the
    # full transcript. Set to "" to disable and use the whole transcript as-is.
    voice_wake_word: str = "hey chef"
    # The one task this test recognizes. If `prompt` matches this (by LLM judgement, so paraphrases are
    # fine), every trajectory in `trajectories_dir` gets played back in sequence.
    task_description: str = "make a sandwich"
    # Folder containing trajectory JSON files saved by `lerobot-teleop-record`.
    trajectories_dir: str = "trajectories"
    # Ollama model used to classify the prompt.
    model: str = DEFAULT_MODEL
    # Ollama server URL.
    host: str = DEFAULT_HOST
    # Playback speed multiplier (2.0 = twice as fast, 0.5 = half speed).
    speed: float = 1.0
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # OpenCV camera device used to photograph the table before running the task (e.g. "/dev/video2" or a
    # plain index like "0"). Run `lerobot-find-cameras opencv` to see what's plugged in. Set to "" to skip
    # the photo/ingredient check and go straight to playback.
    camera: str = "/dev/video2"
    # Ollama vision-language model used to look at the photo and check for ingredients.
    vlm_model: str = DEFAULT_VLM_MODEL
    # Ingredients/items that must be visible on the table before the task is allowed to run.
    ingredients: list[str] = field(default_factory=lambda: ["bread", "cheese", "ham", "lettuce"])
    # If True, refuse to run the task when the vision model reports any ingredient missing. Defaults to
    # False for now (report-only) so the vision check can be exercised on real hardware without needing a
    # fully ingredient-stocked table every time; flip to True once ready to actually gate on it.
    require_ingredients: bool = False
    # Where to save the photo taken for the ingredient check, for debugging/inspection.
    snapshot_path: str = "outputs/vision_checks/latest.jpg"


def capture_photo(camera_id: str) -> NDArray[Any]:
    """Connects to an OpenCV camera just long enough to grab a single RGB frame."""
    index_or_path: int | str = camera_id
    try:
        index_or_path = int(camera_id)
    except ValueError:
        pass

    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=index_or_path, color_mode=ColorMode.RGB))
    camera.connect(warmup=True)
    try:
        return camera.read()
    finally:
        camera.disconnect()


def check_table_for_ingredients(cfg: PlayByPromptConfig) -> bool:
    """Takes a photo with `cfg.camera`, prints what the vision model sees, and checks `cfg.ingredients`.

    Returns True if the task should proceed (either all ingredients are present, or the check is
    non-blocking), False if it should be aborted.
    """
    # Free the text classifier's memory before loading the vision model -- on memory-constrained hardware
    # (e.g. a Jetson's shared CPU/GPU memory pool) keeping both resident at once can exhaust available
    # memory and crash the vision call with a CUDA OOM.
    unload_model(cfg.model, cfg.host)
    ensure_model_available(cfg.vlm_model, cfg.host)

    print(f"\nTaking a photo with camera {cfg.camera!r} to check the table...")
    image = capture_photo(cfg.camera)

    snapshot_path = Path(cfg.snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(image, mode="RGB").save(snapshot_path)
    print(f"Saved photo to {snapshot_path}")

    print(f"Asking the vision model what it sees and checking for: {cfg.ingredients}")
    result, raw_vlm = analyze_table(image, cfg.ingredients, cfg.vlm_model, cfg.host)
    logger.debug("Raw VLM output:\n%s", raw_vlm)

    print(f"\nWhat the robot sees: {result['description']}\n")

    if result["present"]:
        print(f"  Present: {', '.join(result['present'])}")
    if result["missing"]:
        print(f"  Missing: {', '.join(result['missing'])}")

    if result["missing"] and cfg.require_ingredients:
        print(
            f"\nMissing ingredient(s) {result['missing']} -- not starting '{cfg.task_description}'. "
            "Put them on the table and try again, or pass --require_ingredients=false to override."
        )
        return False

    return True


def connect_robot_with_retries(robot: Robot, retries: int = 3, retry_delay_s: float = 2.0) -> None:
    """Connects to `robot`, retrying a few times on transient CAN/motor-bus communication errors (e.g.
    the `motorbridge` control-ack timeouts this hardware occasionally raises) before giving up.

    Does a best-effort disconnect between attempts so a partially-opened bus connection doesn't linger
    and interfere with the next attempt.
    """
    for attempt in range(1, retries + 1):
        try:
            robot.connect()
            return
        except Exception as e:
            if attempt == retries:
                raise
            logger.warning(
                "robot.connect() failed (attempt %d/%d): %s -- retrying in %.1fs...",
                attempt,
                retries,
                e,
                retry_delay_s,
            )
            try:
                robot.disconnect()
            except Exception:
                pass  # nosec B110 -- best-effort cleanup before retrying, original error already logged
            time.sleep(retry_delay_s)


@parser.wrap()
def play_by_prompt(cfg: PlayByPromptConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.voice_list_devices:
        from lerobot.utils.voice_input import list_input_devices

        list_input_devices()
        return

    if cfg.voice:
        if cfg.prompt:
            logger.warning("Both --voice=true and --prompt were given; ignoring --prompt.")
        print("\nGet ready to speak your instruction...")
        prompt = record_and_transcribe(
            model_name=cfg.voice_model,
            language=cfg.voice_language,
            engine=cfg.voice_engine,
            device=cfg.voice_device,
            max_duration=cfg.voice_max_duration,
        )
        print(f"Heard: {prompt!r}")
        if not prompt:
            print("Nothing transcribed -- try again, speak louder/longer, or pass --prompt instead.")
            return

        if cfg.voice_wake_word:
            command = extract_after_wake_word(prompt, cfg.voice_wake_word)
            if command is None:
                print(
                    f"\nDidn't hear the wake phrase {cfg.voice_wake_word!r} -- ignoring. Say "
                    f'"{cfg.voice_wake_word}" followed by your instruction, e.g. "{cfg.voice_wake_word}, '
                    f'make a sandwich".'
                )
                return
            if not command:
                print(
                    f"\nHeard the wake phrase {cfg.voice_wake_word!r} but no instruction after it. Try again."
                )
                return
            print(f"Wake phrase {cfg.voice_wake_word!r} detected.")
            prompt = command

        print(f"Prompt: {prompt!r}")
    elif cfg.prompt:
        prompt = cfg.prompt
    else:
        raise ValueError('Pass either --prompt="..." or --voice=true.')

    paths = list_trajectories(cfg.trajectories_dir)
    if not paths:
        raise FileNotFoundError(
            f"No trajectory files found in '{cfg.trajectories_dir}'. Record one with "
            "`lerobot-teleop-record` first."
        )

    print(f"Asking the local LLM whether {prompt!r} means {cfg.task_description!r}...")
    ensure_server_running(cfg.host)
    ensure_model_available(cfg.model, cfg.host)
    matched, raw = classify_match(prompt, cfg.task_description, cfg.model, cfg.host)
    logger.debug("Raw model output:\n%s", raw)

    if not matched:
        print(f"\n'{prompt}' does not match the known task ('{cfg.task_description}'). Doing nothing.")
        return

    if cfg.camera:
        if not check_table_for_ingredients(cfg):
            return
    else:
        print("\nNo --camera configured, skipping the photo/ingredient check.")

    print(f"\nMatched! Playing back {len(paths)} trajectory file(s) in this order:")
    for p in paths:
        print(f"  - {p.name}")

    robot_action_processor = make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)
    try:
        connect_robot_with_retries(robot)
    except Exception as e:
        print(
            f"\nFailed to connect to the robot after retries: {e}\n"
            "This is a CAN/motor-bus communication error (not related to the prompt/vision check above). "
            "Check that the follower arm is powered on and the CAN adapter/cable is properly connected, "
            "make sure no other process (e.g. lerobot-teleoperate) is still holding the port, then try "
            "again."
        )
        return

    try:
        log_say(f"Running task: {cfg.task_description}", cfg.play_sounds)
        for i, path in enumerate(paths):
            data = load_trajectory(path)
            frames = data["frames"]
            if not frames:
                logger.warning(f"'{path}' has no frames, skipping.")
                continue
            print(f"\n--- [{i + 1}/{len(paths)}] Playing {path.name} ({len(frames)} frames) ---")
            playback_loop(robot, frames, robot_action_processor, cfg.speed)
        log_say("Task complete", cfg.play_sounds)
    except KeyboardInterrupt:
        log_say("Interrupted", cfg.play_sounds)
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    play_by_prompt()


if __name__ == "__main__":
    main()
