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
1. Correction of 4 steps last night:
    - Write an evaluation script and evaluate 3-obj ioi ability on each sliced models
    - Examine the fine-tuned model and find if the 2-object ioi still works there, and whether its heads are still working as before
    - Go through the sliced model and see the transformation inside (both 2obj & 3obj situations)
2. Do it. 