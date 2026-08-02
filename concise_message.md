# YesChef

*Revolute hackathon · Aug 1–2, 2026 · FabLab Kendall, Cambridge MA*

## Inspiration

Everyone's made the joke about telling a robot to make you a sandwich. We wanted to find out how hard it actually is.

Non trivial, it turns out. You have to understand what someone asked for including the parts they left out, you have to know what's actually on the table, and then you have to pick up a slice of tomato without turning it into juice.

## What it does

You say "Hey chef, make me a sandwich" and the arm makes you one. You name how you want it and it uses only those ingredients in the order you want them. So, tomato then cheese gets you tomato then cheese. Ask for a plain sandwich to get the house build.

It starts by taking inventory and proceeds based on the availability.

Everything runs locally on the system.

## How we built it

| Stage | What runs |
|---|---|
| Voice | Wake word: `Hey Chef`, then Whisper AI model takes your voice order on the Jetson |
| Intent | `qwen2.5:3b` in Ollama on the Jetson |
| Vision | `qwen2.5vl:3b`, reads the labels on the packaging |
| Order | String matching in Python |
| Motion | Skill based decomposition executes desired trajectory at 60 Hz|
| Arm | Seeed reBot B601-RS over SocketCAN, TPU compliant gripper |

The organizers provied us with an excellent TPU printed gripper and that flex makes manipulation not destroy a bread.

We used a language model for deciding whether what you said was a sandwich order. Everything after that is string matching. This preserves the ingredients order when you're stacking things and makes down the line process deterministic. The fuzzy question is handled by the model and the deterministic half is taken care of by logic. Perfectly balanced, as all things should be.

Vision works the same way. It identifies if a certain ingredients is present on the table. We not only ask it to reliably recognize food, but we also ask it to find evidence by reading the word printed on the packaging or container.

We didn't go the expected VLA route for a number of reasons. Limited time to collect data, we wouldn't have covered the distribution, and something as simple as a wrong angle of bread or a different texture or a different size would be sufficient to throw off the model. The lightning is also less than ideal. For a precise endeavour such as this, it wasn't something we felt realistic or confident about delivering in 2 days. To overcome this, we collected individual expert trajectories and overfitted an MLP policy on top of them. VLM reasons on selecting the relevant skill based on user input.

## Challenges we ran into

Two 3B models don't fit in 8 GB. Ollama keeps a model warm for five minutes after you use it, so the text model was still sitting in memory when the vision model tried to encode a photo, and we got `cudaMalloc failed` and an HTTP 500 with no useful error until we went digging in the Ollama log. Now we unload the text model before loading the vision one and both run with smaller context windows.

Vision was also too slow to put in front of a judge. Roughly 7 seconds of every call is just encoding the image, so asking two questions cost about 27 seconds warm, and merging them into a single prompt got it to 11. `tegrastats` showed the GPU pinned at 99% the whole time, so that's the hardware talking and not a config mistake.

The arm ate most of a day. The gripper multiplies the leader's angle by six and then clamps the result back into the same range it started in, so only the first 45 degrees of leader travel did anything at all. Wrist roll has a mechanical stop that no config value will move, which we found by widening the software limit and watching nothing change. Rolling the leader past 180 degrees could flip the follower to the opposite limit.

We also faced significant challenges with ingredient handling. You can easily stab a tomato if you come in too steep and it's extremely easy to squish it into pulp before it ever makes it to the sandwich. That gap also shifts with how ripe the tomato is. We had to figure out ingenious (or hacky) ways to make it work. We also snapped a gripper mount when it closed too deep into the cheese container.

Eliminating or mitigating inconsistencies in grasps was also something we addressed. A grasp that works once doesn't work ten times, so spent some time figuring out the approach angles and the exact moment to close the gripper. We ended up building a container for each ingredient so it presents itself the same way on every run. We found staging the food mattered equally or more than tuning the code, which is not what we expected going in.

## Accomplishments that we're proud of

It works end to end, you talk and a sandwich shows up.

We also built a fallback ladder because demos break. Full auto first, then prompt plus trajectories with vision only reporting, then straight playback, and if all of that goes down we drive the arm by hand with the leader and still hand someone a sandwich.

## What we learned

Most of soft-object manipulation happens before any code runs, it's in the gripper and in how you lay the ingredients out.

The other thing we kept relearning is that every time we needed a guarantee the language model was the wrong tool. It's good at "did they mean a sandwich" and bad at "don't change the order", and sorting out which questions were which took a few tries.

A 3B model on a Jetson was plenty for this, memory was the wall and not intelligence.

## What's next

Use the cameras to find the ingredients instead of just confirming they exist so the station doesn't have to be taped down, check that each layer landed before adding the next, and move from overfitted policies to something that survives out of distribution data at inference such as the plate being a few inches off or a bread angled in the other direction.

Sandwiches were the excuse. What we want is to say a sentence out loud and have a physical thing get built with nothing in the loop that needs an internet connection, and we're not there yet.
