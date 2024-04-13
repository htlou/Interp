# My Learning and other stuff in interpretability
## Before the real study
1. Re-learning the introduction of Transformers: finish the three videos in the **What is a Transformer** (along with its code)
    a. proof: Clean_Transformer_Demo_Template.ipynb
    b. Time: ~4h, including the code finishing(have to say some bugs are really annoying)
2. The Exploratory Analysis Demo Tutorial
    a. proof: Exploratory_Analysis_Demo.ipynb
    b. Time: ~2h, Since no coding is needed
3. General paper reading of mech interp(since I already took a surveyon it about a year ago, it's quite easy)
    a. papers: see papers.md
    b. Time: Alas, didn't remember that, because I just kind of going through them in many time pieces

## Entering the real study time
1. Early stage reading and thinking: paper.md
2. Hands-on and experiments: hands_on.md

Note that these two markdowns also serves as some kind of note to myself, so it may contain some offensive words and might be a little nagging.

I believed I explained every files, what I did to them and what I get from them, in these three markdowns (including this one). So it's quite self-containing.

## Conclusion
1. Actually I'm quite reluctant to write this because there's so much to explore, but alas, time is over and I spent more than 10 hours in this experiment and coding(not counting the paper reading and this conclusion writing), and the application table is going to over, so I shall stop right now and arrive at a conclusion.
2. Both hypotheses is partly correct: The finetune process indeed **Transforms** the original circuit, and it **add** new heads (as well as calling MLP layer more often) into this process as the task is getting more complicated.
3. Pity, questions and future
    - **A huge flaw in the experiment**: the corruption only take place in the first of two subjects(if the subject is Jim and John, the answer is only Jim, and the name in the second place never appears in the answer during corruption, for example), so the mechanism of the second place is not well studied.
    - The mechanism of building the new circuit is still under exploration. **How is the new circuit formed?** From my experiments it seems that there is a bit of chaos in the model, for it's adding noise/trying to explore different streategies. But it lacks more ablation and deeper study to explore it.
    - The task is really easy. Actually I don't think it's something bad, as the larger the language model is, the harder for us to design some tasks that the model is capable in its easy form, but fails in harder ones(and the task itself has to be easy enough to be interpreted!). Moreover I think this easy task selection helped me to get slightly different results from the *Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking* paper. For complicated models, most circuits are already be developed during the pretraining, so it's quite good to use small models to look through this process.
    - Lacks theory and mathmatical derivations. I just don't have time and effort to learn such things in 10 hours... and it's too late when I realize that might be fun too.
4. What I'm proud of
    - Come up with this question(though it is already explored several times before), but now my brain is full of training dynamics. What if we combine this dynamics with algorithm designing? I'm quite confident that this helps with the preformance compared to the black-box algorithms.
    - Finishing all these stuff in ~10h, especially the part of fine-tuning models and use Lens to see the dynamics, so much accomplishment!
    - didn't give up, though reach the border of giving up several times.LOL

That's all of my research in totally 12-13 hours, there might be flaws, errors and many not-satifying places, but so far I'm quite happy with it!