from functools import partial
from typing import List, Optional, Union

import einops
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch
from circuitsvis.attention import attention_heads
from fancy_einsum import einsum
from IPython.display import HTML, IFrame
from jaxtyping import Float
from transformers import AutoModelForCausalLM, AutoConfig

import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer
import torch

def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, **kwargs):
    fig = px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    )
    # fig.show()
    return fig


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()

def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
        # Only the final logits are relevant for the answer
        final_logits = logits[:, -1, :]
        answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
        answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
        if per_prompt:
            return answer_logit_diff
        else:
            return answer_logit_diff.mean()

def residual_stack_to_logit_diff(
    residual_stack: Float[torch.Tensor, "components batch d_model"],
    cache: ActivationCache,
) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    return einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        logit_diff_directions,
    ) / len(prompts)

torch.set_grad_enabled(False)
print("Disabled automatic differentiation")

device: torch.device = utils.get_device()

prompt_format = [
    "After John, Mary and Jim went to the grocery store and return, {} gave a bottle of milk to",
    "During the picnic, Billy, Jack, and Ted shared a blanket. {} passed a frisbee to",
    "After Sarah, Tom, and Alex went to the bookstore in the downtown, {} bought a new novel for",
    "Anna, Liam, and Noah decided to buy a cake for someone. {} chose a cake to give to",
]
names = [
    (" Mary", " John", " Jim"),
    (" Billy", " Jack", " Ted"),
    (" Sara", " Tom", " Alex"),
    (" Anna", " Liam", " Noah"),
]


# NBVAL_IGNORE_OUTPUT
model_path = "/mnt/hantao/models/gpt2_sft_2e5_3ep"
config = AutoConfig.from_pretrained(model_path)

filenames_stream = []
filenames_layers = []
for x in range(8, 192, 8):
    # print(i)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        state_dict = torch.load(model_path + f"/pytorch_model_step_{x}.bin")
    )

    model = HookedTransformer.from_pretrained(
        "gpt2",
        hf_model=hf_model,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
     # List of prompts
    prompts = []
    # List of answers, in the format (correct, incorrect)
    answers = []
    # List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
    answer_tokens = []
    for i in range(len(prompt_format)):
        for j in range(3):
            answers.append((names[i][j], names[i][(j - 1)%3], names[i][(j - 2)%3]))
            answer_tokens.append(
                (
                    model.to_single_token(answers[-1][0]),
                    model.to_single_token(answers[-1][1]),
                    model.to_single_token(answers[-1][2]),
                )
            )
            # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
            prompts.append(prompt_format[i].format(answers[-1][1] + " and" + answers[-1][2]))
    answer_tokens = torch.tensor(answer_tokens).to(device)
    print(prompts)
    print(answers)

    for prompt in prompts:
        str_tokens = model.to_str_tokens(prompt)
        print("Prompt length:", len(str_tokens))
        print("Prompt as tokens:", str_tokens)
    tokens = model.to_tokens(prompts, prepend_bos=True)

    # Run the model and cache all activations
    original_logits, cache = model.run_with_cache(tokens)

    print(
        "Per prompt logit difference:",
        logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True)
        .detach()
        .cpu()
        .round(decimals=3),
    )
    original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
    print(
        "Average logit difference:",
        round(logits_to_ave_logit_diff(original_logits, answer_tokens).item(), 3),
    )

    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
    print("Answer residual directions shape:", answer_residual_directions.shape)
    logit_diff_directions = (
        answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    )
    print("Logit difference directions shape:", logit_diff_directions.shape)
    # cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type].
    final_residual_stream = cache["resid_post", -1]
    print("Final residual stream shape:", final_residual_stream.shape)

    final_token_residual_stream = final_residual_stream[:, -1, :]
    # Apply LayerNorm scaling
    # pos_slice is the subset of the positions we take - here the final token of each prompt
    scaled_final_token_residual_stream = cache.apply_ln_to_stack(
        final_token_residual_stream, layer=-1, pos_slice=-1
    )

    average_logit_diff = einsum(
        "batch d_model, batch d_model -> ",
        scaled_final_token_residual_stream,
        logit_diff_directions,
    ) / len(prompts)
    print("Calculated average logit diff:", round(average_logit_diff.item(), 3))
    print("Original logit difference:", round(original_average_logit_diff.item(), 3))

    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
    )
    logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, cache)
    stream_fig = line(
        logit_lens_logit_diffs,
        x=np.arange(model.cfg.n_layers * 2 + 1) / 2,
        hover_name=labels,
        title="Logit Difference From Accumulate Residual Stream",
    )
    savepath = f"assets/stream_{x}.png"
    stream_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    filenames_stream.append(savepath)
    print(f"save stream fig at {savepath}")

    per_layer_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
    per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache)
    layer_fig = line(
        per_layer_logit_diffs, 
        hover_name=labels, 
        title="Logit Difference From Each Layer",
        )
    savepath = f"assets/layer_{x}.png"
    layer_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    filenames_layers.append(savepath)
    print(f"save layer fig at {savepath}")
