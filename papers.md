Claim: This markdown is just my thoughts when going through interp papers before/during the application, so it is not a paper list, it should be... my log in interp? cool.

# General Basis (Might gradually focus on some certain topic)
- A Mathematical Framework for Transformer Circuits (Always fine to start from the tutorial)
- Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (Go further in circuits, read along with the code)
- Finding Neurons in a Haystack: Case Studies with Sparse Probing (Probing basics)
- In-context Learning and Induction Heads (The term of phase change is fascinating)
- Progress measures for grokking via mechanistic interpretability (Interested in Grokking, but it might be too big for 10 hours)
- Universal Neurons in GPT2 Language Models (Generalization of mech interp, but currently it seems not very generalizable...sad)

# Start targeted exploring (Countdown start!)

## Building some global view of problems in mech interp
- Your [200 problem twitter](https://twitter.com/NeelNanda5/status/1608209599844478976)

## Knowledge Editing (Target is to find some applications of Interp)
- Locating and Editing Factual Associations in GPT
- Mass Editing Memory in a Transformer (These two papers are cool because they actually used interp to develop something, which I think is quite a urgent need for current interp community)
- Have read these two papers before, only <0.5h to recall it.

## Random walk from that
- interpreting GPT: the logit lens (Surprised to discover that I've read it before)
- Eliciting Latent Predictions from Transformers with the Tuned Lens (Sounds like logit lens plus, but building a tool like that is also beyond 10 hours)
- Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking (Training dynamics is fantastic! But alas, I have to admit that I really wants to examine the training dynamics as the application. Arrrrgh!)
- <1h, and is really frustrating to face the reality that training dynamic is explored several times...

## Some delves into Circuits
- How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model
- Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla

More than 1h passed after countdown starts... **Tik Tok Tik Tok**

## Stop and think
- It just came to me: The title of the fine-tuning interp paper is *Fine-Tuning Enhances Existing Mechanisms*. What if the model learned a **new** ability in fine-tuning rather then reinforcing **old** ones, for example, chat? 
 - Of course chat might be too comlicated to learn and too difficult to locate, but if I explored the mechanistics in the formation of a certain circuit within a model, although it's not as cool as watching the circuit grows in pretrain stage, it is also attractive.
- Questions: If a new ability is elicited(Since I don't quite believe model actually **acquires** some certain abilities, but rather **elicit** them from their latent knowledge in the fine-tuning stage) from the model in the fine-tuning phase, will it be something like growing a circuit closely similar to existing circuits, or it directly transforms an existing circuit just like I add an "if" sentence in my program in the face of new needs? *i.e.* Observing the generation of a circuit without prerequisites is cool but too hard, and observing this process within a rather intact model will be a step to it.
- Evaluation of the question:
    - fun enough for me to get motivated? Sure, without doubt.
    - valuable? Actually idk, but is it a good step to explore the **transformation** of circuits before reaching the **formation** of it? 
    - can have a early stage exploration in the next ~10h? Ah, another idk... but try my best to do it...
