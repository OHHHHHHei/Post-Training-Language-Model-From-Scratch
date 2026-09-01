from typing import Callable, Literal
from transformers import PreTrainedTokenizerBase

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
        # Entropy is only used for logging. Compute it in small vocabulary
        # chunks so we do not materialize another full [B, T, V] tensor.
        with torch.no_grad():
            token_entropy = torch.zeros(
                all_log_probs.shape[:-1],
                device=all_log_probs.device,
                dtype=torch.float32,
            )
            for log_prob_chunk in all_log_probs.detach().split(1024, dim=-1):
                log_prob_chunk = log_prob_chunk.float()
                token_entropy -= (
                    log_prob_chunk.exp() * log_prob_chunk
                ).sum(dim=-1)
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

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: str = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:


    # raw_rewards_or_advantages:
    # [batch_size] 或 [batch_size, 1]
    advantages = raw_rewards_or_advantages.reshape(-1, 1)

    # policy_log_probs:
    # [batch_size, sequence_length]
    per_token_policy_gradient_loss = (
        -advantages * policy_log_probs
    )

    metadata = {}

    return per_token_policy_gradient_loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:

    # 将 mask 转换成和 loss 一样的数据类型
    mask = mask.to(dtype=per_token_policy_gradient_loss.dtype)

    # 只保留 response token 的 loss
    masked_loss = per_token_policy_gradient_loss * mask

    # 对每个样本的 response token loss 求和
    loss_sum_per_sequence = masked_loss.sum(dim=1)

    # 统计每个样本有多少个 response token
    response_token_count = mask.sum(dim=1).clamp_min(1.0)

    # 每个样本内部，对 response token 求平均,sequence normalization,防止 loss 受到长度影响
    loss_per_sequence = (
        loss_sum_per_sequence / response_token_count
    )

    # 再对 batch 中所有样本求平均
    loss = loss_per_sequence.mean()

    return loss

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    # 清空上一轮反向传播的梯度，切换成训练模式
    optimizer.zero_grad(set_to_none=True)
    model.train()

    # 计算所有 rollout 的 rewards
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )

    # 在完整 batch 上计算 advantages
    advantages, advantages_metadata = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    # 对完整 batch 做 tokenize
    tokenized = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer,
    )

    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    # Tokenization and reward computation happen on CPU, while the policy is
    # normally on GPU. Move every tensor used by the forward pass together.
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    response_mask = response_mask.to(device)
    advantages = advantages.to(device)

    batch_size = len(rollout_responses)
    microbatch_size = batch_size // gradient_accumulation_steps

    total_loss = 0.0
    entropy_sum = 0.0
    entropy_count = 0.0

    for start in range(0, batch_size, microbatch_size):
        end = start + microbatch_size

        input_ids_micro = input_ids[start:end]
        labels_micro = labels[start:end]
        mask_micro = response_mask[start:end]
        advantages_micro = advantages[start:end]

        # 当前 policy 对这些 output 的 log probability
        log_prob_output = get_response_log_probs(
            model=model,
            input_ids=input_ids_micro,
            labels=labels_micro,
            return_token_entropy=True
        )

        policy_log_probs = log_prob_output["log_probs"]
        token_entropy = log_prob_output["token_entropy"]

        # 计算每个 token 的 policy_gradient loss
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=advantages_micro,
            policy_log_probs=policy_log_probs,
            importance_reweighting_method="none",
        )

        # 用 response_mask 把 token loss 转换成 sequence norm 之后的 loss
        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=mask_micro,
            loss_normalization="sequence",
        )

        # 计算 microbatch 占完整 batch 的比例
        weight = (end - start) / batch_size
        # 把当前 microbatch 的平均 loss 按样本比例缩放。
        scaled_loss = microbatch_loss * weight

        # 执行反向传播，梯度会积累到 .grad 里面
        scaled_loss.backward()

        # 记录 loss
        total_loss += scaled_loss.detach().item()

        mask_float = mask_micro.to(
            device=token_entropy.device,
            dtype=token_entropy.dtype,
        )
        entropy_sum += (
            (token_entropy * mask_float).sum().detach().item()
        )
        entropy_count += mask_float.sum().detach().item()

    # 梯度裁剪
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        grad_norm_value = grad_norm.detach().item()
    else:
        grad_norm_value = 0.0


    optimizer.step()

        # 12. 清空梯度，为下一次更新准备
    optimizer.zero_grad(set_to_none=True)

    metadata = {
        **reward_metadata,
        **advantages_metadata,
        "loss": total_loss,
        "grad_norm": grad_norm_value,
        "token_entropy": entropy_sum / max(entropy_count, 1.0),
    }

    return torch.tensor(total_loss), metadata






    
