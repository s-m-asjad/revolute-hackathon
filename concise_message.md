# YesChef

*Revolute hackathon · Aug 1–2, 2026 · FabLab Kendall, Cambridge MA*

## Inspiration

Everyone's made the joke. "Robot, make me a sandwich." We wanted to find out how hard it actually is.

Pretty hard, it turns out. You have to understand what someone asked for, including the parts they left out. You have to know what's actually on the table. And then you have to pick up a slice of tomato without turning it into juice.

## What it does

You say "Hey chef, make me a sandwich." The arm makes one.

Name what you want and it uses only those, in the order you said them. Ask for tomato then cheese and that's the order you get. Ask for a plain sandwich and you get the house build: cucumber, tomato, cheese, lettuce. Bread on the bottom, bread on top.

Before it moves, it photographs the table and tells you what it can see.

All of it runs on the Jetson. No cloud APIs. We didn't pay for anything.

## How we built it

| Stage | What runs |
|---|---|
| Voice | Wake word, then Whisper `tiny.en` on the laptop |
| Intent | `qwen2.5:3b` in Ollama on the Jetson |
| Vision | `qwen2.5vl:3b`, reads the labels on the packaging |
| Order | String matching in Python |
| Motion | Teleop demos saved as 60 Hz JSON, replayed per ingredient |
| Arm | Seeed reBot B601-RS over SocketCAN, TPU compliant gripper |

The gripper does more work than any of our code. The organizers printed it in TPU, and that flex is the only reason a slice of bread survives being picked up.

We used a language model for exactly one thing: deciding whether what you said was a sandwich order. Everything after that is string matching.

That sounds primitive. It's deliberate. Order matters when you're stacking things, and a model that's good at paraphrase is equally good at quietly reordering your list or dropping an item it decided was redundant. So the fuzzy question goes to the model and the exact one stays in code.

Vision works the same way. Ask a 3B model whether there's ham on the table and it will find ham sitting next to a bag of shredded cheese, with total confidence. So we don't ask it to recognize food. We ask it to read the word printed on the package. That costs us recall, and we'll take it, because the failure we care about is the arm reaching into an empty bin.

## Challenges we ran into

Two 3B models don't fit in 16 GB. Ollama keeps a model warm for five minutes after you use it, so the text model was still sitting in memory when the vision model tried to encode a photo. `cudaMalloc failed`, HTTP 500, no useful error until we went digging in the Ollama log. Now we unload the text model before loading the vision one, and both run with smaller context windows.

Vision was also too slow to put in front of a judge. Roughly 7 seconds of every call is just encoding the image, so asking two questions cost about 27 seconds warm. Merging them into a single prompt got it to 11. `tegrastats` showed the GPU pinned at 99% the whole time, so that's the hardware talking, not a config mistake.

The arm ate most of a day. The gripper multiplies the leader's angle by six and then clamps the result back into the same range it started in, so only the first 45 degrees of leader travel did anything at all. Wrist roll has a mechanical stop that no config value will move, which we learned by widening the software limit and watching nothing change. Rolling the leader past 180 degrees could flip the follower to the opposite limit.

And then the part nobody warns you about. A grasp that works once doesn't work ten times. Most of our hours went into approach angles and the exact moment to close the gripper. We ended up building a container for each ingredient so it presents itself the same way on every run. Staging the food mattered more than tuning the code, which is not what we expected going in.

## Accomplishments that we're proud of

It works end to end. You talk, a sandwich shows up.

We also built a fallback ladder, because demos break. Full auto first. If vision gets weird, prompt plus trajectories with vision only reporting. If the prompt path breaks, straight playback. If all of that breaks, we drive the arm by hand with the leader and still hand someone a sandwich.

## What we learned

Most of soft-object manipulation happens before any code runs. It's in the gripper, and in how you lay the ingredients out.

The other thing we kept relearning: every time we needed a guarantee, the language model was the wrong tool. It's good at "did they mean a sandwich." It's bad at "don't change the order." Sorting out which questions were which took a few tries.

A 3B model on a Jetson was plenty for this. Memory was the wall, not intelligence.

## What's next

Use the cameras to find the ingredients instead of just confirming they exist, so the station doesn't have to be taped down. Check that each layer landed before adding the next. Move from replaying recorded motions to something that survives the plate being a few inches off.

Sandwiches were the excuse. What we want is to say a sentence out loud and have a physical thing get built, with nothing in the loop that needs an internet connection. We're not there yet.
