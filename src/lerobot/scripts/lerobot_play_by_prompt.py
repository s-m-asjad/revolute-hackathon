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
ingredients/items are actually on the table -- this is printed to the console either way. It then plays
back a sandwich-assembly sequence built from `--trajectories_dir`: the bread trajectory, then whichever
fillable ingredients (cucumber, tomato, cheese, lettuce) were requested, then the bread trajectory again
(see "Sandwich assembly order" below). If the prompt doesn't match, nothing happens -- the robot is never
touched.

Sandwich assembly order:

- Plain requests ("make a sandwich", "make me a sandwich") play EVERY configured ingredient (that has a
  recorded trajectory file) in the fixed default order set by `--ingredient_order` (cucumber, tomato,
  cheese, lettuce), bracketed by the bread trajectory at the start and end.
- Requests that name specific ingredients (e.g. "make a sandwich with cheese and tomato") only play those,
  in the order they were said (e.g. "tomato then cheese" plays tomato before cheese) -- still bracketed by
  bread at the start and end.
- The bread trajectory always plays first and last. The end uses `--bread2_trajectory` if that file has
  been recorded; otherwise it reuses `--bread_trajectory` (the same trajectory the start used).
- An ingredient with no recorded trajectory file yet (e.g. lettuce, until it's recorded) is skipped with a
  printed warning instead of failing the whole run.

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
from lerobot.utils.ingredient_matching import extract_requested_ingredients
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
from lerobot.utils.trajectory_io import load_trajectory
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
    # fine), the sandwich-assembly trajectory sequence below gets played back.
    task_description: str = "make a sandwich"
    # Folder containing trajectory JSON files saved by `lerobot-teleop-record`. Filenames below are
    # resolved relative to this directory unless they're absolute paths.
    trajectories_dir: str = "trajectories"
    # Trajectory played first (the bottom bread slice).
    bread_trajectory: str = "first_bread.json"
    # Trajectory played last (the top bread slice). Falls back to `bread_trajectory` if this file hasn't
    # been recorded yet, so the sandwich still gets "closed" with the same bread trajectory used to start.
    bread2_trajectory: str = "rsbread_second.json"
    # Maps each fillable ingredient to its trajectory file. For a plain "make a sandwich" request, every
    # ingredient here (that has an existing trajectory file) plays in `ingredient_order`. If the prompt
    # names specific ingredients instead, only those play, in the order the user said them.
    ingredient_trajectories: dict[str, str] = field(
        default_factory=lambda: {
            "cucumber": "rscucumber.json",
            "tomato": "srstomato.json",
            "cheese": "rscheese.json",
            "lettuce": "rslettuce.json",
        }
    )
    # Default playback order for the fillable ingredients when the prompt doesn't name specific ones.
    ingredient_order: list[str] = field(default_factory=lambda: ["cucumber", "tomato", "cheese", "lettuce"])
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


def _resolve_trajectory_path(trajectories_dir: Path, filename: str) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else trajectories_dir / path


def _require_bread_trajectory(cfg: PlayByPromptConfig) -> None:
    """Raises `FileNotFoundError` if the bread trajectory is missing, without printing anything -- used
    to fail fast, before bothering Ollama/the camera, without pre-announcing the sandwich plan for a
    prompt that might not even match the known task yet.
    """
    bread_path = _resolve_trajectory_path(Path(cfg.trajectories_dir), cfg.bread_trajectory)
    if not bread_path.is_file():
        raise FileNotFoundError(
            f"Bread trajectory '{bread_path}' not found. Record it with `lerobot-teleop-record` first, "
            "or point --bread_trajectory at the right file."
        )


def build_trajectory_plan(cfg: PlayByPromptConfig, prompt: str) -> list[tuple[str, Path]]:
    """Builds the ordered list of (label, trajectory_path) steps for `prompt`.

    Always starts with the bread trajectory and ends with the bread trajectory (using
    `cfg.bread2_trajectory` for the end if that file exists, otherwise reusing `cfg.bread_trajectory`).
    In between: every ingredient in `cfg.ingredient_order` that has an existing trajectory file, unless
    `prompt` explicitly names specific ingredients -- in which case only those play, in the order they
    were named. Ingredients with no recorded trajectory file yet are skipped with a printed warning
    rather than aborting the whole run.

    Raises `FileNotFoundError` if the bread trajectory itself is missing, since there's no sandwich
    without it.
    """
    trajectories_dir = Path(cfg.trajectories_dir)

    bread_path = _resolve_trajectory_path(trajectories_dir, cfg.bread_trajectory)
    if not bread_path.is_file():
        raise FileNotFoundError(
            f"Bread trajectory '{bread_path}' not found. Record it with `lerobot-teleop-record` first, "
            "or point --bread_trajectory at the right file."
        )

    bread2_path = _resolve_trajectory_path(trajectories_dir, cfg.bread2_trajectory)
    if not bread2_path.is_file():
        print(
            f"No second-bread trajectory at '{bread2_path}' yet -- closing the sandwich with the first "
            f"bread trajectory ('{bread_path.name}') again instead."
        )
        bread2_path = bread_path

    requested = extract_requested_ingredients(prompt, list(cfg.ingredient_trajectories.keys()))
    if requested is not None:
        print(f"Specific ingredient(s) requested, in this order: {', '.join(requested) or '(none)'}")
        ingredient_names = requested
    else:
        default_order = ", ".join(cfg.ingredient_order)
        print(f"No specific ingredients named -- using the full default order: {default_order}")
        ingredient_names = cfg.ingredient_order

    plan: list[tuple[str, Path]] = [("bread", bread_path)]
    for name in ingredient_names:
        filename = cfg.ingredient_trajectories.get(name)
        if filename is None:
            print(f"  Skipping '{name}': no trajectory file configured for it.")
            continue
        path = _resolve_trajectory_path(trajectories_dir, filename)
        if not path.is_file():
            print(f"  Skipping '{name}': trajectory file '{path}' doesn't exist yet.")
            continue
        plan.append((name, path))
    plan.append(("bread", bread2_path))

    return plan


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

    # Fail fast (before touching Ollama/the camera) if the bread trajectory -- required for every run --
    # hasn't been recorded yet.
    _require_bread_trajectory(cfg)

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

    plan = build_trajectory_plan(cfg, prompt)
    print(f"\nMatched! Playing back {len(plan)} trajectory step(s) in this order:")
    for label, path in plan:
        print(f"  - {label}: {path.name}")

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
        for i, (label, path) in enumerate(plan):
            data = load_trajectory(path)
            frames = data["frames"]
            if not frames:
                logger.warning(f"'{path}' has no frames, skipping.")
                continue
            print(f"\n--- [{i + 1}/{len(plan)}] Playing {label} ({path.name}, {len(frames)} frames) ---")
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
