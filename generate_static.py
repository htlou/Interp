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
    fig = px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    )
    return fig


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
    fig = px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    )
    return fig

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

def patch_residual_component(
    corrupted_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    clean_cache,
):
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component


def normalize_patched_logit_diff(patched_logit_diff):
    # Subtract corrupted logit diff to measure the improvement, divide by the total improvement from clean to corrupted to normalise
    # 0 means zero change, negative means actively made worse, 1 means totally recovered clean performance, >1 means actively *improved* on clean performance
    return (patched_logit_diff - corrupted_average_logit_diff) / (
        original_average_logit_diff - corrupted_average_logit_diff
    )

def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    head_index,
    clean_cache,
):
    corrupted_head_vector[:, :, head_index, :] = clean_cache[hook.name][
        :, :, head_index, :
    ]
    return corrupted_head_vector

def patch_head_pattern(
    corrupted_head_pattern: Float[torch.Tensor, "batch head_index query_pos d_head"],
    hook,
    head_index,
    clean_cache,
):
    corrupted_head_pattern[:, head_index, :, :] = clean_cache[hook.name][
        :, head_index, :, :
    ]
    return corrupted_head_pattern

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
    
    per_head_residual, labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache)
    per_head_logit_diffs = einops.rearrange(
        per_head_logit_diffs,
        "(layer head_index) -> layer head_index",
        layer=model.cfg.n_layers,
        head_index=model.cfg.n_heads,
    )
    head_fig = imshow(
        per_head_logit_diffs,
        labels={"x": "Head", "y": "Layer"},
        title="Logit Difference From Each Head",
    )
    savepath = f"assets/head_{x}.png"
    head_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    print(f"save head fig at {savepath}")

    # Let's Corrupt!
    corrupted_prompts = []
    for i in range(0, len(prompts), 3):
        # Add second prompt in the trio first
        corrupted_prompts.append(prompts[(i + 1) % len(prompts)])
        # Then add the first prompt
        corrupted_prompts.append(prompts[i % len(prompts)])
        # Finally, add the third prompt, wrapping around the list safely
        corrupted_prompts.append(prompts[(i + 2) % len(prompts)])
        

    # Proceed with tokenization and model running as before
    corrupted_tokens = model.to_tokens(corrupted_prompts, prepend_bos=True)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens, return_type="logits")
    corrupted_average_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
    print("Corrupted Average Logit Diff:", round(corrupted_average_logit_diff.item(), 2))
    print("Clean Average Logit Diff:", round(original_average_logit_diff.item(), 2))
    print(model.to_string(corrupted_tokens))

    patched_residual_stream_diff = torch.zeros(
        model.cfg.n_layers, tokens.shape[1], device=device, dtype=torch.float32
    )
    for layer in range(model.cfg.n_layers):
        for position in range(tokens.shape[1]):
            hook_fn = partial(patch_residual_component, pos=position, clean_cache=cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("resid_pre", layer), hook_fn)],
                return_type="logits",
            )
            patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)

            patched_residual_stream_diff[layer, position] = normalize_patched_logit_diff(
                patched_logit_diff
            )
    prompt_position_labels = [
        f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(tokens[0]))
    ]
    patch_res_fig = imshow(
        patched_residual_stream_diff,
        x=prompt_position_labels,
        title="Logit Difference From Patched Residual Stream",
        labels={"x": "Position", "y": "Layer"},
    )
    savepath = f"assets/patch_res_{x}.png"
    patch_res_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    print(f"save patch residual stream fig at {savepath}")

    patched_attn_diff = torch.zeros(
        model.cfg.n_layers, tokens.shape[1], device=device, dtype=torch.float32
    )
    patched_mlp_diff = torch.zeros(
        model.cfg.n_layers, tokens.shape[1], device=device, dtype=torch.float32
    )
    for layer in range(model.cfg.n_layers):
        for position in range(tokens.shape[1]):
            hook_fn = partial(patch_residual_component, pos=position, clean_cache=cache)
            patched_attn_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("attn_out", layer), hook_fn)],
                return_type="logits",
            )
            patched_attn_logit_diff = logits_to_ave_logit_diff(
                patched_attn_logits, answer_tokens
            )
            patched_mlp_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("mlp_out", layer), hook_fn)],
                return_type="logits",
            )
            patched_mlp_logit_diff = logits_to_ave_logit_diff(
                patched_mlp_logits, answer_tokens
            )

            patched_attn_diff[layer, position] = normalize_patched_logit_diff(
                patched_attn_logit_diff
            )
            patched_mlp_diff[layer, position] = normalize_patched_logit_diff(
                patched_mlp_logit_diff
            )

    patch_attn_fig = imshow(
        patched_attn_diff,
        x=prompt_position_labels,
        title="Logit Difference From Patched Attention Layer",
        labels={"x": "Position", "y": "Layer"},
    )

    savepath = f"assets/patch_attn_{x}.png"
    patch_attn_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    print(f"save patch attention layer fig at {savepath}")

    patch_mlp_fig = imshow(
        patched_mlp_diff,
        x=prompt_position_labels,
        title="Logit Difference From Patched MLP Layer",
        labels={"x": "Position", "y": "Layer"},
    )
    savepath = f"assets/patch_mlp_{x}.png"
    patch_mlp_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    print(f"save patch mlp layer fig at {savepath}")

    patched_head_z_diff = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
    )
    for layer in range(model.cfg.n_layers):
        for head_index in range(model.cfg.n_heads):
            hook_fn = partial(patch_head_vector, head_index=head_index, clean_cache=cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("z", layer, "attn"), hook_fn)],
                return_type="logits",
            )
            patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)

            patched_head_z_diff[layer, head_index] = normalize_patched_logit_diff(
                patched_logit_diff
            )

    patch_heado_fig = imshow(
        patched_head_z_diff,
        title="Logit Difference From Patched Head Output",
        labels={"x": "Head", "y": "Layer"},
    )

    savepath = f"assets/patch_heado_{x}.png"
    patch_heado_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    print(f"save patch head output fig at {savepath}")

    patched_head_v_diff = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
    )
    for layer in range(model.cfg.n_layers):
        for head_index in range(model.cfg.n_heads):
            hook_fn = partial(patch_head_vector, head_index=head_index, clean_cache=cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("v", layer, "attn"), hook_fn)],
                return_type="logits",
            )
            patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)

            patched_head_v_diff[layer, head_index] = normalize_patched_logit_diff(
                patched_logit_diff
            )
    patch_vdiff_fig = imshow(
        patched_head_v_diff,
        title="Logit Difference From Patched Head Value",
        labels={"x": "Head", "y": "Layer"},
    )
    savepath = f"assets/patch_vdiff_{x}.png"
    patch_vdiff_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    print(f"save patch head value fig at {savepath}")

    head_labels = [
        f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ]
    out_vs_value_fig = scatter(
        x=utils.to_numpy(patched_head_v_diff.flatten()),
        y=utils.to_numpy(patched_head_z_diff.flatten()),
        xaxis="Value Patch",
        yaxis="Output Patch",
        caxis="Layer",
        hover_name=head_labels,
        color=einops.repeat(
            np.arange(model.cfg.n_layers), "layer -> (layer head)", head=model.cfg.n_heads
        ),
        range_x=(-0.5, 0.5),
        range_y=(-0.5, 0.5),
        title="Scatter plot of output patching vs value patching",
    )
    savepath = f"assets/out_vs_value_{x}.png"
    out_vs_value_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    print(f"save out_vs_value fig at {savepath}")


    patched_head_attn_diff = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
    )
    for layer in range(model.cfg.n_layers):
        for head_index in range(model.cfg.n_heads):
            hook_fn = partial(patch_head_pattern, head_index=head_index, clean_cache=cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(utils.get_act_name("attn", layer, "attn"), hook_fn)],
                return_type="logits",
            )
            patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)

            patched_head_attn_diff[layer, head_index] = normalize_patched_logit_diff(
                patched_logit_diff
            )

    head_pattern_fig = imshow(
        patched_head_attn_diff,
        title="Logit Difference From Patched Head Pattern",
        labels={"x": "Head", "y": "Layer"},
    )
    head_labels = [
        f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ]
    out_vs_attn_fig = scatter(
        x=utils.to_numpy(patched_head_attn_diff.flatten()),
        y=utils.to_numpy(patched_head_z_diff.flatten()),
        hover_name=head_labels,
        xaxis="Attention Patch",
        yaxis="Output Patch",
        title="Scatter plot of output patching vs attention patching",
    )

    savepath = f"assets/head_pattern_{x}.png"
    head_pattern_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    print(f"save head_pattern fig at {savepath}")

    savepath = f"assets/out_vs_attn_{x}.png"
    out_vs_attn_fig.write_image(savepath)  # 使用Plotly的write_image方法保存图像
    print(f"save out_vs_attn fig at {savepath}")