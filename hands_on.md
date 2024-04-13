- Questions: If a new ability is elicited(Since I don't quite believe model actually **acquires** some certain abilities, but rather **elicit** them from their latent knowledge in the fine-tuning stage) from the model in the fine-tuning phase, will it be something like growing a circuit closely similar to existing circuits, or it directly transforms an existing circuit just like I add an "if" sentence in my program in the face of new needs? *i.e.* Observing the generation of a circuit without prerequisites is cool but too hard, and observing this process within a rather intact model will be a step to it.
- Evaluation of the question:
    - fun enough for me to get motivated? Sure, without doubt.
    - valuable? Actually idk, but is it a good step to explore the **transformation** of circuits before reaching the **formation** of it? 
    - can have a early stage exploration in the next ~10h? Ah, another idk... but try my best to do it...
## Prerequisites & Plans
1. Two similar domains, and GPT-2 is good in one but bad in another.explore time!
    - Three object commplicated reasoning: John has a ball. John passes the ball to Mary. Mary passes the ball to Jim. The ball is now in the hand of {Jim} model failed in this mission
    - Two object reasoning: John and Mary come into the room, John went out of the room. The person in the room is {Mary} model succeeded
    - Three object reasoning: John and Mary and Jim come into the room, Jim went out of the room. John went out of the room. The person in the room is {Mary} model failed
    - Three object reasoning: John and Mary and Jim come into the room, John and Jim went out of the room. The person in the room is {Mary} model failed
    - Two object reasoning with noise: John and Mary come into the room. John went out of the room. Jim went out of the room. The person in the room is {Mary} The model answered John
    - Greater-than + reasoning: John buys 1 bottle of milk. Mary buys 10 bottles of milk. The person who buys more bottles of milk is {Mary} model failed
    - two object, multi step: John comes into the room. Mary comes into the room. John went out of the room. The person in the room is {Mary} model failed in that.
    - three object ioi: "After John, Mary and Jim went to the store, Mary and John gave a bottle of milk to" {Jim} model failed, output John
    - After John and Mary went to the store. Jim stay at the house. Mary and John gave a bottle of milk to {Jim} the model also failed.

2. Decide to take: 3 objects ioi
    - Obeserve that 2 objects ioi is quite easy for models. First I may get a 2 objects ioi circuit, then identify it in 3 objects ioi situation. It's easy to figure that 3 objects circuit will fail at first, but after I SFT the model in a 3 objects ioi dataset, I reckon something might change.

3. Steps:
    - Build a 3-item ioi dataset using GPT API.
    - SFT this dataset on GPT2-small.
    - Observing the result.
    - (Same time) re-run the note book to activate my memory in ioi circuit

## Coding Stage starts!
1. dataset: I clone my gpt4_eval repository to annotate a dataset.
    - prompt: system_prompt.py
    - output: output.json
2. SFT: Use the pku safe-rlhf repository to perform sft, but met a bug when loading the sft model to HookedTransformer: *RuntimeError: The size of tensor a (50257) must match the size of tensor b (50258) at non-singleton dimension 0*
3. It's 3h now, try to debug now
4. holy sh*t I forgot to delete the pad token in the sft config
5. bug 2 fixed, but the output is terrible... freaking overfit!
6. set lr to 2e-05, but it just don't learn from that...so the current situation is either overfit or doesn't fit at all.
7. shift to a bigger model, gpt2-xl & gpt-2 large
8. discovered that gpt2-xl can finish this task naturally, so narrow to gpt2-medium(cannot finish this task)
9. 5h and discover I'm completely f*cked up, either the ability of 3 object ioi is limited by the model size and cannot be elicited by finetuning, or I failed in my sft(It might because the output length). In this case I will just copy my SFT code here and move on to somewhere else. Take a rest and start tomorrow.

10. For god's sake I succeeded!!! The question is one token output is too short, and the model always end in learning just output an <|endoftext|>, but I'm not that talented to go through the finetuning repo and change the training logic in the left <10h time, so I pad the completion of the dataset: piece["completion"] = (" "+piece["completion"])*20 (Compare the output_new.json and output.json!) This magic code helped me control the length and now the finetuned gpt2-small model **can** perform 3 object ioi task with accuracy 10/10(just test them one by one)! start analyzing at once.

11. Copy some of the code from Exploratory_Analysis_Demo.ipynb, and partly proved that the heads used for 2 object ioi also plays a role in the 3 object ioi, so it (to some extent) prove that the circuits are actually doing **Transformation** rather than a raw **Formation**. So the next step I think 3 things is quite important (with diminishing importance):
    - Examine the fine-tuned model and find if the 2-object ioi still works there, and whether its heads are still working as before
    - Go through the sliced model and see the transformation inside (both 2obj & 3obj situations)
    - try the 3-object problem on the original gpt2-small model and see if I can find the trace of heads in the 3-object ioi heads working in the 2-obj problem (I guess I won't have time for this though, just let it be)
12. Urggh its really late now, just some more experiment and I shall go sleeping, keep the clock at 7 hour, and start off tomorrow.

13. start off just playing the 2-obj ioi on fine-tuned model in the ..._Playground.ipynb It isn't much surprise that the pattern of attention head still exists in 2-obj situations, that give more credit to te statement that the circuit is being **transformed** into a new one, which is capable for both the old task and the newly-learned ones.
14. I believe from this work I sort of discovered that the circuit is evolving in complicated tasks in order to extract more latent knowledge from the model, or to say that the model is actually use its latent knowledge to help circuits evolve? I can't say right now, but it's such a good hypothesis!

## Going Deep
1. Correction of 3 steps last night:
    - Write an evaluation script and evaluate 3-obj ioi ability on each sliced models
    - Examine the fine-tuned model and find if the 2-object ioi still works there, and whether its heads are still working as before
    - Go through the sliced model and see the transformation inside (both 2obj & 3obj situations)
2. Eval the models and ckpts:
    - runs the sample.py to get a 50 item test set
    - write the eval.py and runs it to eval all the ckpt of the model, get:
```
ckpt8, win rate 38.0%
ckpt16, win rate 46.0%
ckpt24, win rate 52.0%
ckpt32, win rate 54.0%
ckpt40, win rate 57.99999999999999%
ckpt48, win rate 54.0%
ckpt56, win rate 57.99999999999999%
ckpt64, win rate 64.0%
ckpt72, win rate 64.0%
ckpt80, win rate 66.0%
ckpt88, win rate 66.0%
ckpt96, win rate 68.0%
ckpt104, win rate 70.0%
ckpt112, win rate 74.0%
ckpt120, win rate 78.0%
ckpt128, win rate 84.0%
ckpt136, win rate 88.0%
ckpt144, win rate 90.0%
ckpt152, win rate 90.0%
ckpt160, win rate 90.0%
ckpt168, win rate 90.0%
ckpt176, win rate 92.0%
ckpt184, win rate 94.0%

need to mention that the ckpt 0 (original gpt2) also has a win rate of 38%
```

3. It seems that this experiment proved my guess again: when the model is familliar with some tasks (*i.e.* they fully developed a circuit to do it), finetuning then on a similar(but harder, they can't finish alone before) task won't be the situation like the grokking paper. Instead, the win rate just smoothly grow to nearly 100%, so the circuit is transformed smoothly too.

4. So let's go interpreting the circuit. some of the process afterwards is directly shown in explore.ipynb, but I'll try my best to copy everything here.

5. Logit Diff: Compared with the logit difference of the original model, these two plots show that most parts of the process is quite the same, although at layer 10 is displays some difference: its logit difference continues to grow(although decreases at 11 remains the same) instead of beginning to decrease.That suggestst that the difference might happens in layer 10. Let's make a gif graph of the sliced models. See generate_static.py and generate_gif.py, which get the image of layer & stream logit diff.
6. Get leyer_plots.gif and stream_plots.gif, didn't fit the tick because I'm in a huge lack of time...it's already about 9 hours
7. From these two graphs, I observe that the layer 10 is the key layer. among all layers, 10 is the most active during finetuning. It fluctuates all along the finetuning, and develop from non-active(generally in original pattern, layer 11 is a layer of decreasing differnence) to active(a growing differnence other than layer 9)
8. does it means that it is something like a residual learning, *i.e.* layer 10 learns the diff of current circuit in layer 9 and targeted circuit? idk, see the attention for more details
9. L10H0 and L10H7 calls for my attention. its activation rate has the biggest changing over the time. see the attention details of them.
10. Gazing at the attention analysis is not enough, since I'm not wise enough to directly get the circuit diff from the complicated graph, decide to move on.
