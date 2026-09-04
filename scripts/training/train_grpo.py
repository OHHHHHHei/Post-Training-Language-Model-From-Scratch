"""在 GSM8K 上训练 GRPO 的简化脚本。"""

import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from post_training.gsm8k_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from post_training.grpo import (
    get_response_log_probs,
    grpo_train_step,
    tokenize_prompt_and_output,
)
from post_training.vllm_utils import VLLMServer


# 根据实验需要修改这些配置。
MODEL_ID = "allenai/OLMo-2-0425-1B"
PROMPT_CONFIGS = {
    "r1_zero": {
        "path": "post_training/prompts/gsm8k/r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
    },
    "question_only": {
        "path": "post_training/prompts/gsm8k/question_only.prompt",
        "reward_fn": question_only_reward_fn,
        "stop": None,
    },
    "three_shot": {
        "path": "post_training/prompts/gsm8k/r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
    },
}
PROMPT_NAME = os.environ.get("GRPO_PROMPT", "r1_zero")
PROMPT_CONFIG = PROMPT_CONFIGS[PROMPT_NAME]
PROMPT_PATH = PROMPT_CONFIG["path"]
REWARD_FN = PROMPT_CONFIG["reward_fn"]
STOP_STRINGS = PROMPT_CONFIG["stop"]
TRAIN_PATH = "data/gsm8k/train.jsonl"
VAL_PATH = "data/gsm8k/test.jsonl"

N_TRAIN_EXAMPLES = 6400
N_VAL_EXAMPLES = 1024
NUM_ROLLOUT_STEPS = int(os.environ.get("GRPO_STEPS", "50"))
ROLLOUT_BATCH_SIZE = 256
GROUP_SIZE = 8
VARIANT = os.environ.get("GRPO_VARIANT", "standard")
OFF_POLICY_METHODS = {
    "offpolicy_naive": "none",
    "offpolicy_noclip": "noclip",
    "offpolicy_clip": "grpo",
    "offpolicy_gspo": "gspo",
}
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get(
        "GRPO_GRAD_ACCUM",
        "2" if VARIANT in OFF_POLICY_METHODS else "64",
    )
)
LEARNING_RATE = float(os.environ.get("GRPO_LR", "1e-5"))
TEMPERATURE = 1.0
MAX_TOKENS = 512
MAX_GRAD_NORM = 1.0
EVAL_EVERY = 10
SEED = int(os.environ.get("GRPO_SEED", "42"))

# 常数归一化变体使用固定分母。
NORMALIZATION_CONSTANT = int(
    os.environ.get(
        "GRPO_NORMALIZATION_CONSTANT",
        str(ROLLOUT_BATCH_SIZE * MAX_TOKENS),
    )
)

VARIANT_CONFIGS = {
    "standard": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "normalization_constant": None,
    },
    "grpo_constant": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
    "dr_grpo": {
        "baseline": "mean",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
    "rft": {
        "baseline": "none",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
    "maxrl": {
        "baseline": "mean",
        "advantage_normalizer": "mean",
        "loss_normalization": "constant",
        "normalization_constant": NORMALIZATION_CONSTANT,
    },
}

if VARIANT in OFF_POLICY_METHODS:
    VARIANT_CONFIG = {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "normalization_constant": None,
        "importance_reweighting_method": OFF_POLICY_METHODS[VARIANT],
        "cliprange": {
            "offpolicy_clip": 0.2,
            "offpolicy_gspo": 3e-4,
        }.get(VARIANT),
    }
else:
    VARIANT_CONFIG = {
        **VARIANT_CONFIGS[VARIANT],
        "importance_reweighting_method": "none",
        "cliprange": None,
    }

IS_OFF_POLICY = VARIANT in OFF_POLICY_METHODS
TRAIN_BATCH_SIZE = ROLLOUT_BATCH_SIZE // 32 if IS_OFF_POLICY else ROLLOUT_BATCH_SIZE
NUM_TRAIN_UPDATES_PER_ROLLOUT = ROLLOUT_BATCH_SIZE // TRAIN_BATCH_SIZE

if IS_OFF_POLICY:
    DEFAULT_OUTPUT_DIR = Path("experiments/off_policy") / VARIANT.removeprefix(
        "offpolicy_"
    )
else:
    DEFAULT_OUTPUT_DIR = Path("experiments/on_policy") / VARIANT
if PROMPT_NAME != "r1_zero":
    DEFAULT_OUTPUT_DIR = Path("experiments/prompt_ablation") / PROMPT_NAME

OUTPUT_DIR = Path(os.environ.get("GRPO_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))

POLICY_DEVICE = "cuda:0"
VLLM_GPU = 2
VLLM_PORT = 8010


def load_examples(path: str, limit: int) -> list[dict[str, str]]:
    examples = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            answer = item["answer"].split("####", 1)[1].strip()
            examples.append({"question": item["question"], "answer": answer})
    return examples[:limit]


def make_prompts(template: str, examples: list[dict[str, str]]):
    prompts = [template.replace("{question}", item["question"]) for item in examples]
    answers = [item["answer"] for item in examples]
    return prompts, answers


def generate(server, prompts: list[str], seed: int, n: int = 1):
    sampling_params = {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "n": n,
        "seed": seed,
    }
    if STOP_STRINGS is not None:
        sampling_params["stop"] = STOP_STRINGS
        sampling_params["include_stop_str_in_output"] = True
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=64,
    )
    return [completion.text for completion in completions]


def compute_old_log_probs(
    policy,
    tokenizer,
    prompts: list[str],
    responses: list[str],
):
    """计算一组 rollout 在参数更新前的旧策略 log-prob。"""
    tokenized = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer,
    )
    input_ids = tokenized["input_ids"].to(POLICY_DEVICE)
    labels = tokenized["labels"].to(POLICY_DEVICE)

    policy.eval()
    with torch.no_grad():
        log_prob_output = get_response_log_probs(
            model=policy,
            input_ids=input_ids,
            labels=labels,
        )
    return log_prob_output["log_probs"].cpu()


def average_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """将同一批 rollout 的多次训练更新指标取平均。"""
    keys = {key for metrics in metrics_list for key in metrics}
    return {
        key: sum(metrics.get(key, 0.0) for metrics in metrics_list)
        / len(metrics_list)
        for key in keys
    }


def evaluate(
    server,
    tokenizer,
    template: str,
    examples: list[dict[str, str]],
    step: int,
):
    prompts, answers = make_prompts(template, examples)
    responses = generate(server, prompts, SEED + 10000 + step)
    scores = [REWARD_FN(response, answer) for response, answer in zip(responses, answers)]
    response_tokenized = tokenizer(responses, add_special_tokens=False)
    average_response_length = sum(
        len(ids) for ids in response_tokenized["input_ids"]
    ) / len(responses)
    return {
        "val_reward": sum(score["reward"] for score in scores) / len(scores),
        "val_format_reward": sum(score["format_reward"] for score in scores) / len(scores),
        "val_answer_reward": sum(score["answer_reward"] for score in scores) / len(scores),
        "val_avg_response_length": average_response_length,
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    with open(PROMPT_PATH, "r", encoding="utf-8") as file:
        prompt_template = file.read()

    train_data = load_examples(TRAIN_PATH, N_TRAIN_EXAMPLES)
    val_data = load_examples(VAL_PATH, N_VAL_EXAMPLES)
    random.shuffle(train_data)

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(POLICY_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # OLMo tokenizer 没有 pad token 时，使用 eos token 进行 batch padding。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.use_cache = False

    # 优化器只更新训练模型，vLLM 通过权重同步获得最新参数。
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    server = VLLMServer(
        model_id=MODEL_ID,
        host="localhost",
        port=VLLM_PORT,
        gpu=VLLM_GPU,
        seed=SEED,
        logging_level="INFO",
        gpu_memory_utilization=0.8,
    )
    # 启动 vLLM，后续用它批量生成 rollout。
    server.start()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / f"config_seed{SEED}.json").open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(
            {
                "variant": VARIANT,
                "seed": SEED,
                "model_id": MODEL_ID,
                "prompt_name": PROMPT_NAME,
                "prompt_path": PROMPT_PATH,
                "n_train_examples": N_TRAIN_EXAMPLES,
                "n_val_examples": N_VAL_EXAMPLES,
                "num_rollout_steps": NUM_ROLLOUT_STEPS,
                "rollout_batch_size": ROLLOUT_BATCH_SIZE,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "num_train_updates_per_rollout": NUM_TRAIN_UPDATES_PER_ROLLOUT,
                "group_size": GROUP_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "max_grad_norm": MAX_GRAD_NORM,
                "eval_every": EVAL_EVERY,
                "variant_config": VARIANT_CONFIG,
            },
            config_file,
            indent=2,
        )
    metrics_file = (OUTPUT_DIR / f"metrics_seed{SEED}.jsonl").open(
        "w", encoding="utf-8"
    )
    rollouts_file = (OUTPUT_DIR / f"rollouts_seed{SEED}.jsonl").open(
        "w", encoding="utf-8"
    )

    try:
        server.init_weight_sync(POLICY_DEVICE)
        prompts_per_batch = ROLLOUT_BATCH_SIZE // GROUP_SIZE

        for step in range(NUM_ROLLOUT_STEPS):
            batch = random.sample(train_data, prompts_per_batch)
            prompts, answers = make_prompts(prompt_template, batch)
            repeated_prompts = [p for p in prompts for _ in range(GROUP_SIZE)]
            repeated_answers = [a for a in answers for _ in range(GROUP_SIZE)]

            # 每批 rollout 生成前，把当前训练模型的权重同步到 vLLM。
            server.sync_policy_weights(policy)
            responses = generate(server, prompts, SEED + step, n=GROUP_SIZE)

            # 定期保存 rollout 样例，方便检查模型输出。
            if (step + 1) % 40 == 0:
                for prompt, response, answer in zip(
                    repeated_prompts, responses, repeated_answers
                ):
                    rollouts_file.write(
                        json.dumps(
                            {
                                "step": step + 1,
                                "prompt": prompt,
                                "response": response,
                                "ground_truth": answer,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                rollouts_file.flush()

            # 旧策略概率必须在第一次参数更新前固定下来。
            old_log_probs_by_group = []
            if VARIANT_CONFIG["importance_reweighting_method"] != "none":
                for update_index in range(NUM_TRAIN_UPDATES_PER_ROLLOUT):
                    start = update_index * TRAIN_BATCH_SIZE
                    end = start + TRAIN_BATCH_SIZE
                    old_log_probs_by_group.append(compute_old_log_probs(
                        policy=policy,
                        tokenizer=tokenizer,
                        prompts=repeated_prompts[start:end],
                        responses=responses[start:end],
                    ))

            # 在线策略时这里只有一次更新；离线策略时同一批 rollout
            # 被拆成 32 个 group，连续进行 32 次参数更新。
            update_metrics = []
            for update_index in range(NUM_TRAIN_UPDATES_PER_ROLLOUT):
                start = update_index * TRAIN_BATCH_SIZE
                end = start + TRAIN_BATCH_SIZE
                old_log_probs = (
                    old_log_probs_by_group[update_index]
                    if old_log_probs_by_group
                    else None
                )
                _, metrics_for_update = grpo_train_step(
                    model=policy,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                    max_grad_norm=MAX_GRAD_NORM,
                    reward_fn=REWARD_FN,
                    repeated_prompts=repeated_prompts[start:end],
                    rollout_responses=responses[start:end],
                    repeated_ground_truths=repeated_answers[start:end],
                    group_size=GROUP_SIZE,
                    old_log_probs=old_log_probs,
                    **VARIANT_CONFIG,
                )
                update_metrics.append(metrics_for_update)

            train_metrics = average_metrics(update_metrics)

            metrics = {
                "step": step + 1,
                "variant": VARIANT,
                "seed": SEED,
                **train_metrics,
            }
            if (step + 1) % EVAL_EVERY == 0:
                server.sync_policy_weights(policy)
                metrics.update(
                    evaluate(
                        server,
                        tokenizer,
                        prompt_template,
                        val_data,
                        step + 1,
                    )
                )
                print(metrics)
            else:
                print(
                    f"step={step + 1} "
                    f"train_reward={train_metrics['mean_reward']:.4f} "
                    f"loss={train_metrics['loss']:.4f}"
                )

            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
    finally:
        metrics_file.close()
        rollouts_file.close()
        server.stop()


if __name__ == "__main__":
    main()
