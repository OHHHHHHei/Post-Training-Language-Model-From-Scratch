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

    # 分别对提示词和回答进行分词，然后拼接它们的 ID。
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

    # 将拼接后的序列填充到统一长度。
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
    # logits 的形状为：[批次大小，序列长度，词表大小]

    # 沿词表维度计算 log softmax
    all_log_probs = torch.log_softmax(logits, dim=-1)

    # 取出 labels 对应 token 的对数概率
    label_indices = labels.unsqueeze(-1)

    # 得到真实 label 对应的对数概率
    log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=label_indices,
    ).squeeze(-1)

    result = {
        "log_probs": log_probs
    }

    if return_token_entropy:
        # 熵只用于日志记录。按较小的词表分块计算，避免额外构造完整的
        # [B, T, V] 张量。
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
    # group_size 必须是正数
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    if baseline not in {"mean", "none"}:
        raise ValueError(f"Unsupported baseline: {baseline}")

    if advantage_normalizer not in {"std", "none", "mean"}:
        raise ValueError(
            f"Unsupported advantage_normalizer: {advantage_normalizer}"
        )

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
    # 形状为：[num_groups, 1]
    group_means = grouped_rewards.mean(
        dim=1,
        keepdim=True,
    )

    # baseline = mean 时减去组内平均 reward；baseline = none 时保留原始 reward。
    if baseline == "mean":
        centered_rewards = grouped_rewards - group_means
    else:
        centered_rewards = grouped_rewards

    # 计算每组 reward 的标准差
    # 形状为：[num_groups, 1]
    group_stds = grouped_rewards.std(
        dim=1,
        keepdim=True,
    )

    if advantage_normalizer == "std":
        advantages = centered_rewards / (group_stds + advantage_eps)
    elif advantage_normalizer == "none":
        advantages = centered_rewards
    else:
        advantages = centered_rewards / (group_means + advantage_eps)

    # 恢复成一维，方便后面的策略梯度损失使用
    advantages = advantages.reshape(-1)

    # 元数据只用于日志，不参与反向传播
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

    # raw_rewards_or_advantages 的形状：
    # [batch_size] 或 [batch_size, 1]
    advantages = raw_rewards_or_advantages.reshape(-1, 1)

    # policy_log_probs 的形状：
    # [batch_size, sequence_length]
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length)")
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("advantages and policy_log_probs batch sizes must match")

    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -advantages * policy_log_probs
        metadata = {}
        return per_token_policy_gradient_loss, metadata

    if importance_reweighting_method not in {"noclip", "grpo", "gspo"}:
        raise NotImplementedError(
            f"Unsupported importance_reweighting_method: {importance_reweighting_method}"
        )
    if old_log_probs is None:
        raise ValueError(
            "old_log_probs is required for off-policy importance reweighting"
        )
    if old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("old_log_probs and policy_log_probs must have the same shape")

    # 在对数空间中计算每个 token 的重要性比率。旧策略保持不变，不能接收梯度。
    log_ratio = policy_log_probs - old_log_probs.detach()

    if importance_reweighting_method == "gspo":
        if response_mask is None:
            raise ValueError("response_mask is required for GSPO")
        if response_mask.shape != policy_log_probs.shape:
            raise ValueError(
                "response_mask and policy_log_probs must have the same shape"
            )
        if cliprange is None:
            raise ValueError("cliprange is required for GSPO clipping")
        if cliprange < 0:
            raise ValueError("cliprange must be non-negative")

        # GSPO 在回答 token 上平均 log-ratio，再取指数得到几何平均权重。
        # 这样避免了直接连乘 token-level 权重造成的数值不稳定。
        mask = response_mask.to(
            device=policy_log_probs.device,
            dtype=policy_log_probs.dtype,
        )
        response_token_count = mask.sum(dim=1).clamp_min(1.0)
        sequence_log_ratio = (
            (log_ratio * mask).sum(dim=1) / response_token_count
        )
        sequence_ratio = torch.exp(sequence_log_ratio)

        sequence_advantages = advantages.squeeze(1)
        clipped_sequence_ratio = torch.clamp(
            sequence_ratio,
            min=1.0 - cliprange,
            max=1.0 + cliprange,
        )
        unclipped_objective = sequence_advantages * sequence_ratio
        clipped_objective = sequence_advantages * clipped_sequence_ratio
        sequence_loss = -torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        # 后续聚合函数仍以 token-level 张量为接口，因此将同一个
        # sequence-level loss 广播到该序列的所有位置。
        per_token_policy_gradient_loss = sequence_loss.unsqueeze(1).expand_as(
            policy_log_probs
        )

        clipped_sequences = (
            ((sequence_advantages > 0) & (sequence_ratio > 1.0 + cliprange))
            | ((sequence_advantages < 0) & (sequence_ratio < 1.0 - cliprange))
        )
        metadata = {
            "clip_fraction": clipped_sequences.to(
                dtype=policy_log_probs.dtype
            ).mean().detach()
        }
        return per_token_policy_gradient_loss, metadata

    importance_ratio = torch.exp(log_ratio)

    if importance_reweighting_method == "noclip":
        # 代理目标函数为 A * w_t。梯度下降需要最小化它的相反数，
        # 因此每个 token 的损失为 -A * w_t。
        per_token_policy_gradient_loss = -advantages * importance_ratio
        metadata = {}
        return per_token_policy_gradient_loss, metadata

    if cliprange is None:
        raise ValueError("cliprange is required for GRPO clipping")
    if cliprange < 0:
        raise ValueError("cliprange must be non-negative")

    clipped_ratio = torch.clamp(
        importance_ratio,
        min=1.0 - cliprange,
        max=1.0 + cliprange,
    )
    unclipped_objective = advantages * importance_ratio
    clipped_objective = advantages * clipped_ratio
    per_token_policy_gradient_loss = -torch.minimum(
        unclipped_objective,
        clipped_objective,
    )

    # 统计 PPO/GRPO 目标函数选择裁剪分支的 token 比例。
    clipped_tokens = (
        ((advantages > 0) & (importance_ratio > 1.0 + cliprange))
        | ((advantages < 0) & (importance_ratio < 1.0 - cliprange))
    )
    if response_mask is None:
        clip_fraction = clipped_tokens.to(dtype=policy_log_probs.dtype).mean()
    else:
        mask = response_mask.to(
            device=policy_log_probs.device,
            dtype=policy_log_probs.dtype,
        )
        clip_fraction = (
            clipped_tokens.to(dtype=policy_log_probs.dtype) * mask
        ).sum() / mask.sum().clamp_min(1.0)

    metadata = {"clip_fraction": clip_fraction.detach()}

    return per_token_policy_gradient_loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:

    if loss_normalization not in {"sequence", "constant"}:
        raise ValueError(
            f"Unsupported loss_normalization: {loss_normalization}"
        )

    # 将 mask 转换成和损失一样的数据类型
    mask = mask.to(dtype=per_token_policy_gradient_loss.dtype)

    # 只保留回答部分 token 的损失
    masked_loss = per_token_policy_gradient_loss * mask

    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError(
                "normalization_constant is required for constant normalization"
            )
        if normalization_constant <= 0:
            raise ValueError("normalization_constant must be positive")
        return masked_loss.sum() / normalization_constant

    # 对每个样本的回答部分 token 损失求和
    loss_sum_per_sequence = masked_loss.sum(dim=1)

    # 统计每个样本有多少个回答部分 token
    response_token_count = mask.sum(dim=1).clamp_min(1.0)

    # 对每个样本内部的回答部分 token 求平均，执行序列归一化，
    # 防止损失受到回答长度影响。
    loss_per_sequence = (
        loss_sum_per_sequence / response_token_count
    )

    # 再对批次中的所有样本求平均
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

    # 清空上一轮反向传播的梯度，并切换到训练模式
    optimizer.zero_grad(set_to_none=True)
    model.train()

    # 计算所有 rollout 的 reward
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )

    # 在完整批次上计算 advantage
    advantages, advantages_metadata = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    # 对完整批次进行分词
    tokenized = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer,
    )

    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    batch_size = len(rollout_responses)
    if batch_size == 0:
        raise ValueError("rollout_responses must not be empty")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            "batch_size must be divisible by gradient_accumulation_steps"
        )

    # advantage 为 0 的样本梯度也为 0。在模型前向计算前将它们过滤掉，
    # 但归一化时仍保留原始批次大小。
    active_indices = torch.nonzero(advantages != 0, as_tuple=True)[0]
    active_batch_size = active_indices.numel()
    if active_batch_size < batch_size:
        input_ids = input_ids.index_select(0, active_indices)
        labels = labels.index_select(0, active_indices)
        response_mask = response_mask.index_select(0, active_indices)
        advantages = advantages.index_select(0, active_indices)
        if old_log_probs is not None:
            old_log_probs = old_log_probs.index_select(
                0,
                active_indices.to(old_log_probs.device),
            )

    # 分词和 reward 计算在 CPU 上完成，策略模型通常在 GPU 上运行。
    # 将前向计算所需的所有张量一起移动到模型所在设备。
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    response_mask = response_mask.to(device)
    advantages = advantages.to(device)
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    microbatch_size = batch_size // gradient_accumulation_steps

    total_loss = 0.0
    entropy_sum = 0.0
    entropy_count = 0.0
    clip_fraction_sum = 0.0
    clip_fraction_count = 0

    for start in range(0, active_batch_size, microbatch_size):
        end = min(start + microbatch_size, active_batch_size)

        input_ids_micro = input_ids[start:end]
        labels_micro = labels[start:end]
        mask_micro = response_mask[start:end]
        advantages_micro = advantages[start:end]
        old_log_probs_micro = (
            None if old_log_probs is None else old_log_probs[start:end]
        )

        # 当前策略对这些输出的对数概率
        log_prob_output = get_response_log_probs(
            model=model,
            input_ids=input_ids_micro,
            labels=labels_micro,
            return_token_entropy=True
        )

        policy_log_probs = log_prob_output["log_probs"]
        token_entropy = log_prob_output["token_entropy"]

        # 计算每个 token 的策略梯度损失
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=advantages_micro,
            policy_log_probs=policy_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs_micro,
            cliprange=cliprange,
            response_mask=mask_micro,
        )

        if "clip_fraction" in loss_metadata:
            clip_fraction_sum += (
                loss_metadata["clip_fraction"].detach().item()
                * (end - start)
            )
            clip_fraction_count += end - start

        # 使用 response_mask 聚合 token 损失。
        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=mask_micro,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        if loss_normalization == "sequence":
            # 序列归一化后的 microbatch 损失是当前有效样本的平均值，
            # 因此需要在这里恢复成完整批次的平均值。
            weight = (end - start) / batch_size
            scaled_loss = microbatch_loss * weight
        else:
            # 常数归一化已经使用全局分母，因此各个 microbatch
            # 的损失直接相加即可。
            scaled_loss = microbatch_loss

        # 执行反向传播，梯度会累积到 .grad 中
        scaled_loss.backward()

        # 记录损失
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
    if max_grad_norm is not None and active_batch_size > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        grad_norm_value = grad_norm.detach().item()
    else:
        grad_norm_value = 0.0


    optimizer.step()

    # 清空梯度，为下一次更新准备
    optimizer.zero_grad(set_to_none=True)

    metadata = {
        **reward_metadata,
        **advantages_metadata,
        "loss": total_loss,
        "grad_norm": grad_norm_value,
        "token_entropy": entropy_sum / max(entropy_count, 1.0),
    }
    if clip_fraction_count > 0:
        metadata["clip_fraction"] = (
            clip_fraction_sum / clip_fraction_count
        )

    return torch.tensor(total_loss), metadata






    
