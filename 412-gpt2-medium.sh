#!/usr/bin/env bash
#
# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
# ==============================================================================

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

# -------------------need to change-------------
export LOGLEVEL="INFO"
# export WANDB_API_KEY="0e77f7c02e33b86269ca2123964b9fefcf9c1a7a"
MODEL_NAME_OR_PATH="gpt2-medium"
OUTPUT_DIR="/mnt/hantao/models/gpt2-medium_sft_5e5"
# DATASET="test.json"
DATASET="/mnt/hantao/evaluation/gpt4_eval/output.json"
# -------------------need to change------------

unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="none"
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--model_name_or_path=*)
			MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
		--output_dir=*)
			OUTPUT_DIR="${arg#*=}"
			;;
		--hostfile)
			HOSTFILE="$1"
			shift
			;;
		--hostfile=*)
			HOSTFILE="${arg#*=}"
			;;
		--zero_stage)
			ZERO_STAGE="$1"
			shift
			;;
		--zero_stage=*)
			ZERO_STAGE="${arg#*=}"
			;;
		--offload)
			OFFLOAD="$1"
			shift
			;;
		--offload=*)
			OFFLOAD="${arg#*=}"
			;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

# if [[ -z "${WANDB_API_KEY}" ]]; then
# 	export WANDB_MODE="offline"
# fi

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

DEEPSPEED_ARGS=()
if [[ -n "${HOSTFILE+x}" ]]; then
	DEEPSPEED_ARGS+=("--hostfile" "${HOSTFILE}")
fi
DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

deepspeed "${DEEPSPEED_ARGS[@]}" \
	--module safe_rlhf.finetune \
	--train_datasets interp::${DATASET} \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--trust_remote_code True \
	--epochs 3 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 4 \
	--gradient_checkpointing \
	--learning_rate 5e-5 \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.0 \
	--seed 42 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project aligner-base \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True \
	--save_interval 8 \
	--save_16bit
