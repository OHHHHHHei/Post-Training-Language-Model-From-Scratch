# 1. 启动 / 使用 vLLM 加载 OLMo-2-0425-1B
# 2. 读取 GSM8K jsonl
# 3. 从每条数据里拿 question 和 ground-truth answer
# 4. 读取 prompt 模板
# 5. 把 question 填进 prompt，得到 prompt string
# 6. 按 prompt 类型设置 generation 参数
# 7. 用 vLLM 生成 response
# 8. 选择对应 reward_fn
# 9. 用 response + ground_truth answer 打分
# 10. 统计三类 category
# 11. 保存样例和 metrics
# 12. 写 PDF 要求的 commentary
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.vllm_utils import VLLMServer

PROMPT_NAME = "r1_zero"  # 可选 "question_only", "r1_zero", "r1_zero_three_shot_gsm8k"
NUM_EXAMPLES = 1319
DATA_PATH = "data/gsm8k/test.jsonl"
MODEL_ID = "allenai/OLMo-2-0425-1B"
PORT = 8010
GPU = 2
SEED = 42

prompt_configs = {
    "question_only": {
        "prompt_path": "cs336_alignment/prompts/question_only.prompt",
        "reward_fn": question_only_reward_fn,
        "stop": None,
    },
    "r1_zero": {
        "prompt_path": "cs336_alignment/prompts/r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
    },
    "r1_zero_three_shot_gsm8k": {
        "prompt_path": "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
    },
}

config = prompt_configs[PROMPT_NAME]
prompt_path = config["prompt_path"]
reward_fn = config["reward_fn"]
stop = config["stop"]

with open(DATA_PATH, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f.readlines()]
    question = [item["question"] for item in test_data]
    answer = [item["answer"] for item in test_data]
    final_answer = [answer[i].split("####")[1].strip() for i in range(len(answer))]  # 取出 ground-truth answer


with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_template = f.read()
    prompts = [prompt_template.replace("{question}", question[i]) for i in range(len(question))]  # 填充 question

# 定义 vllm server
server = VLLMServer(model_id=MODEL_ID, host="localhost", port=PORT, gpu=GPU, seed=SEED, logging_level="info")
server.start()
try:
    sampling_params = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": SEED,
    }
    if stop is not None:
        sampling_params["stop"] = stop
        sampling_params["include_stop_str_in_output"] = True

    completions = server.generate_completions(prompts=prompts[:NUM_EXAMPLES], sampling_params=sampling_params)
    
    counts = {"category_1": 0, "category_2": 0, "category_3": 0, "other": 0}
    for i, completion in enumerate(completions):
        response = completion.text
        ground_truth = final_answer[i]
        score = reward_fn(response, ground_truth)
        
        if score["format_reward"] == 1 and score["answer_reward"] == 1:
            category = 1
            counts["category_1"] += 1
        elif score["format_reward"] == 1 and score["answer_reward"] == 0:
            category = 2
            counts["category_2"] += 1
        elif score["format_reward"] == 0 and score["answer_reward"] == 0:
            category = 3
            counts["category_3"] += 1
        else:
            category = "other"
            counts["other"] += 1
    print(counts)
finally:
    server.stop()