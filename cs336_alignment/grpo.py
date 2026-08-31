from typing import Callable, Literal

import torch

MODEL_ID = "allenai/OLMo-2-0425-1B"

def tokenize_prompt_and_output(
    prompt_strs: list[str], 
    output_strs: list[str], 
    tokenizer,
) -> dict[str, torch.Tensor]:
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    # Tokenize prompts and responses separately, then concatenate their IDs.
    prompt_tokenized = tokenizer(
        prompt_strs,
        add_special_tokens=False
    )
    output_tokenized = tokenizer(
        output_strs,
        add_special_tokens=False,
    )

    prompt_ids = prompt_tokenized["input_ids"]
    output_ids = output_tokenized["input_ids"]
    full_ids = [
        prompt_id + output_id
        for prompt_id, output_id in zip(prompt_ids, output_ids)
    ]

    # Pad the concatenated sequences to a common length.
    tokenized = tokenizer.pad(
        {"input_ids": full_ids},
        padding=True,
        return_tensors="pt",
    )

    full_input_ids = tokenized["input_ids"]
    full_attention_mask = tokenized["attention_mask"]
    prompt_lengths = torch.tensor(
        [len(ids) for ids in prompt_ids],
        dtype=torch.long
    )

    # 输入跟标签要错一位
    input_ids = full_input_ids[:, :-1]
    labels = full_input_ids[:, 1:]

    sequence_length = labels.shape[1]
    label_positions = torch.arange(sequence_length).unsqueeze(0) + 1
    full_lengths = full_attention_mask.sum(dim=1).unsqueeze(1)

    response_mask = (
        (label_positions >= prompt_lengths.unsqueeze(1))
        & (label_positions < full_lengths)
    ).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    # 前向计算
    logits = model(input_ids).logits
    # logits shape: [batch_size, sequence_length, vocab_size]

    # 对词表维度做 log softmax
    all_log_probs = torch.log_softmax(logits, dim=-1)

    # 取出 labels 对应 token 的 log probability
    label_indices = labels.unsqueeze(-1)

    # 得到取出真实 label 的概率
    log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=label_indices,
    ).squeeze(-1)

    result = {
        "log_probs": log_probs
    }

    if return_token_entropy:
        probs = all_log_probs.exp()
        token_entropy = -(probs * all_log_probs).sum(dim=-1)
        result["token_entropy"] = token_entropy

    return result

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length"
        )

    reward_outputs = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses,
            repeated_ground_truths,
        )
    ]

    raw_rewards = torch.tensor(
        [output["reward"] for output in reward_outputs],
        dtype=torch.float32,
    )
    format_rewards = torch.tensor(
        [output["format_reward"] for output in reward_outputs],
        dtype=torch.float32,
    )
    answer_rewards = torch.tensor(
        [output["answer_reward"] for output in reward_outputs],
        dtype=torch.float32,
    )

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": format_rewards.mean().item(),
        "mean_answer_reward": answer_rewards.mean().item(),
    }

    return raw_rewards, metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    # 目前只实现标准 GRPO
    if baseline != "mean":
        raise NotImplementedError(
            "Only baseline='mean' is supported for standard GRPO"
        )

    if advantage_normalizer != "std":
        raise NotImplementedError(
            "Only advantage_normalizer='std' is supported for standard GRPO"
        )

    # group_size 必须是正数
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    # 把输入整理成一维
    raw_rewards = raw_rewards.reshape(-1)

    # rollout 数量必须能够被 group_size 整除
    if raw_rewards.numel() % group_size != 0:
        raise ValueError(
            "The number of rewards must be divisible by group_size"
        )

    # [num_rollouts] -> [num_groups, group_size]
    grouped_rewards = raw_rewards.reshape(-1, group_size)

    # 计算每个问题对应的一组 reward 的平均值
    # shape: [num_groups, 1]
    group_means = grouped_rewards.mean(
        dim=1,
        keepdim=True,
    )

    # baseline = mean
    # 每个回答减去所在 group 的平均 reward
    centered_rewards = grouped_rewards - group_means

    # 计算每组 reward 的标准差
    # shape: [num_groups, 1]
    group_stds = grouped_rewards.std(
        dim=1,
        keepdim=True,
    )

    # 标准 GRPO 的 advantage
    advantages = centered_rewards / (
        group_stds + advantage_eps
    )

    # 恢复成一维，方便后面的 policy-gradient loss 使用
    advantages = advantages.reshape(-1)

    # metadata 只用于日志，不参与反向传播
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "mean_group_std": group_stds.mean().item(),
    }

    return advantages, metadata

    
