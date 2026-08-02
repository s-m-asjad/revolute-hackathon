# YesChef

*Revolute hackathon · Aug 1–2, 2026 · FabLab Kendall, Cambridge MA*

## Inspiration

"Make me a sandwich" is the canonical joke request for a robot. It's a joke because it's hard: parse what someone wants and doesn't want, check what's actually on the table, then grip things that bend, slip, and tear. We built it.

## What it does

Say "Hey chef, make me a sandwich" and a 6-DOF arm builds one.

Name your fillings and it plays only those, **in the order you said them** — "tomato then cheese" stacks differently from "cheese then tomato." Ask for a plain sandwich and it builds the house version: cucumber, tomato, cheese, lettuce. Bread goes down first and last. Before it moves, it photographs the station and reports which ingredients it can actually see.

Everything runs on-device. No paid APIs, no cloud.

## How we built it

| Stage | Implementation |
|---|---|
| Voice | Wake word → Whisper `tiny.en` on the laptop |
| Intent | `qwen2.5:3b` via Ollama on the Jetson — is this a sandwich order? |
| Vision | `qwen2.5vl:3b` photographs the station, reads ingredient labels |
| Order parsing | Deterministic string matching — **not** an LLM |
| Motion | Teleop demos recorded as 60 Hz JSON trajectories, replayed per ingredient |
| Hardware | reBot B601-RS over SocketCAN · Jetson Orin Nano · USB cams · TPU compliant gripper |

Two decisions did most of the work.

**Parsing is dumb on purpose.** Sequence is semantic here — it's a stack, and the order determines the sandwich. A paraphrase-tolerant model is free to silently reorder or drop an item. A string matcher isn't. We gave the LLM the fuzzy job (did they ask for a sandwich?) and kept the exact job in code.

**Vision reads text, not food.** A 3B VLM asked to identify ham will confidently find ham next to a bag of shredded cheese. We prompt it to read the ingredient name printed on the packaging instead. That trades recall for precision, which is the right trade when a false positive means reaching into an empty bin.

## Challenges we ran into

**Two 3B models don't fit.** Ollama keeps a model warm for 5 minutes after use. With the text and vision models both resident, the Jetson's shared 16 GB pool ran out mid-encode — `cudaMalloc failed`, HTTP 500. Fixed by unloading the text model (`keep_alive=0`) before loading the vision model, and shrinking both context windows to 1024/2048.

**Vision was too slow to demo.** Each call pays ~7 s just to encode the photo, so describing the scene and checking ingredients as two calls cost ~27 s warm. Merging them into one prompt, answered in one line, got it to ~11 s. `tegrastats` showed the GPU pinned at 99% — genuinely compute-bound, not misconfigured.

**The arm had opinions.** The gripper multiplies the leader's angle by 6 and then clamps to the same range, so only the first 45° of leader travel did anything. Wrist roll has a hard mechanical stop that no config change moves — we found that the expensive way, by widening the limit first. Rolling the leader past 180° could flip the follower to the opposite limit, fixed with an unwrap patch. CAN handoff after a teleop session needed connect retries.

**Replay is harder than demonstration.** A grasp that works once doesn't survive ten runs. Most of our time went into approach angles, speeds, and grasp timing — plus custom containers so each ingredient presents itself to the gripper the same way every time.

## Accomplishments that we're proud of

Voice to sandwich, end to end, on free local models. Wake word, speech-to-text, intent, vision, order parsing, and compliant manipulation of soft food all had to work in sequence — and the layers land without drops.

We also built a failure ladder that keeps the demo alive: full auto, then prompt plus trajectories with vision report-only, then trajectory playback, then live teleop. The last rung still makes a sandwich in front of a judge.

## What we learned

Most of soft-object manipulation is solved before any code runs — in the compliant gripper and in how you stage the ingredients. The rest is earned back through patient trajectory tuning.

The other lesson was where *not* to put a model. Language models are good at ambiguity and bad at guarantees. Every place we needed a guarantee — ingredient order, label identity — we took the model out.

A 3B model on a Jetson is enough for intent and perception at this scale. The constraint was memory and latency, not capability.

## What's next

Use the cameras to locate ingredients rather than just confirm them, so the station doesn't have to be perfectly staged. Verify each layer landed before placing the next. Move from replaying trajectories to per-ingredient ACT policies that generalize across placements. Streaming speech-to-text with barge-in.

The pipeline isn't really about sandwiches — it's voice to perception to manipulation, for anyone who wants to ask a robot in plain language to assemble something from parts.
