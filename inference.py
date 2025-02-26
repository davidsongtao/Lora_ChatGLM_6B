import time
import torch
import numpy as np
from typing import Dict, List, Any, Tuple

from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList
from glm_config import ProjectConfig


class FirstTokenTimingCallback(StoppingCriteria):
    """记录第一个token生成时间的回调"""

    def __init__(self):
        self.first_token_time = None
        self.start_time = time.time()
        self.tokens_generated = 0
        self.token_timestamps = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.tokens_generated = input_ids.shape[1]
        current_time = time.time()

        # 记录第一个新生成的token的时间
        if self.first_token_time is None and self.tokens_generated > input_ids.shape[1] - 1:
            self.first_token_time = current_time - self.start_time
            self.token_timestamps.append((1, current_time - self.start_time))
        else:
            # 记录每个token的时间戳
            new_tokens = self.tokens_generated - (len(self.token_timestamps) + input_ids.shape[1] - 1)
            if new_tokens > 0:
                self.token_timestamps.append((new_tokens, current_time - self.start_time))

        return False  # 不停止生成


def inference(
        model,
        tokenizer,
        instruction: str,
        sentence: str,
        max_new_tokens: int = 300
) -> Dict[str, Any]:
    """
    模型 inference 函数，增加性能指标计算。

    Args:
        model: 模型
        tokenizer: 分词器
        instruction (str): 指令
        sentence (str): 输入句子
        max_new_tokens (int): 最大生成token数

    Returns:
        Dict: 包含结果和性能指标的字典
    """
    with torch.no_grad():
        input_text = f"Instruction: {instruction}\n"
        if sentence:
            input_text += f"Input: {sentence}\n"
        input_text += f"Answer: "

        # 计算输入的token数量
        batch = tokenizer(input_text, return_tensors="pt")
        input_token_len = batch["input_ids"].shape[1]

        # 创建回调来记录第一个token的时间
        timing_callback = FirstTokenTimingCallback()

        # 记录总体开始时间
        overall_start = time.time()

        # 使用回调进行生成
        out = model.generate(
            input_ids=batch["input_ids"].to(pc.device),
            max_new_tokens=max_new_tokens,
            temperature=0,
            stopping_criteria=StoppingCriteriaList([timing_callback])
        )

        # 计算总体耗时
        total_time = time.time() - overall_start

        # 解码输出
        out_text = tokenizer.decode(out[0])
        answer = out_text.split('Answer: ')[-1]

        # 计算生成的token数量
        output_token_len = out.shape[1] - input_token_len

        # 性能指标
        metrics = {
            "answer": answer,
            "first_token_latency": timing_callback.first_token_time,  # 第一个token的响应时间
            "tokens_per_second": output_token_len / total_time,  # 每秒生成的token数
            "total_output_tokens": output_token_len,  # 总输出token数
            "total_time": total_time,  # 总耗时
            "token_timestamps": timing_callback.token_timestamps,  # token时间戳记录
        }

        return metrics


def calculate_throughput(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算平均吞吐量和其他聚合指标

    Args:
        metrics_list: 多次推理的指标列表

    Returns:
        Dict: 聚合指标
    """
    avg_first_token = np.mean([m["first_token_latency"] for m in metrics_list])
    avg_tokens_per_second = np.mean([m["tokens_per_second"] for m in metrics_list])

    # 计算95%置信区间
    ftl_95ci = 1.96 * np.std([m["first_token_latency"] for m in metrics_list]) / np.sqrt(len(metrics_list))
    tps_95ci = 1.96 * np.std([m["tokens_per_second"] for m in metrics_list]) / np.sqrt(len(metrics_list))

    return {
        "avg_first_token_latency": avg_first_token,
        "avg_tokens_per_second": avg_tokens_per_second,
        "ftl_95ci": ftl_95ci,
        "tps_95ci": tps_95ci,
        "total_samples": len(metrics_list)
    }


if __name__ == '__main__':
    from rich.console import Console
    from rich.table import Table

    console = Console()

    trained_model = "/root/autodl-fs/trained_models/chatglm_6b/sop_chatglm_6b"
    max_new_tokens = 300
    pc = ProjectConfig()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            trained_model,
            trust_remote_code=True,
            revision="main"
        )

        model = AutoModel.from_pretrained(
            trained_model,
            trust_remote_code=True,
            revision="main"
        ).half().to(pc.device)
        console.print('模型加载成功...', style="green")
    except Exception as e:
        console.print(e, style="red")
        console.print('加载模型失败，请检查模型路径是否正确', style="red")
        exit()

    samples = [
        {
            'instruction': "现在你是一个非常厉害的SPO抽取器。",
            "input": "下面这句中包含了哪些三元组，用json列表的形式回答，不要输出除json外的其他答案。\n\n73获奖记录人物评价：黄磊是一个特别幸运的演员，拍第一部戏就碰到了导演陈凯歌，而且在他的下一部电影《夜半歌声》中演对手戏的张国荣、吴倩莲、黎明等都是著名的港台演员。",
        },
        {
            'instruction': "你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。",
            "input": "下面句子中的主语是什么类别，输出成列表形式。\n\n第N次入住了，就是方便去客户那里哈哈。还有啥说的"
        }
    ]

    console.print('开始推理评测...', style="bold blue")
    console.print('=' * 50)

    # 预热模型
    console.print("预热模型中...", style="yellow")
    _ = inference(model, tokenizer, "你好", "世界", max_new_tokens=10)

    # 存储所有样本的指标
    all_metrics = []

    # 样本循环
    for i, sample in enumerate(samples):
        console.print(f"样本 {i + 1}/{len(samples)}", style="bold")
        console.print(f"人类提问--> {sample['input']}")

        # 执行推理并获取指标
        metrics = inference(
            model,
            tokenizer,
            sample['instruction'],
            sample['input'],
            max_new_tokens=max_new_tokens
        )

        all_metrics.append(metrics)

        # 显示结果
        console.print(f"模型回答 --> {metrics['answer']}")

        # 显示性能指标
        table = Table(title=f"样本 {i + 1} 性能指标")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")

        table.add_row("首token响应时间", f"{metrics['first_token_latency']:.4f} 秒")
        table.add_row("每秒token数", f"{metrics['tokens_per_second']:.2f} tokens/s")
        table.add_row("总token数", f"{metrics['total_output_tokens']} tokens")
        table.add_row("总耗时", f"{metrics['total_time']:.4f} 秒")

        console.print(table)
        console.print('=' * 50)

    # 计算并显示聚合指标
    summary = calculate_throughput(all_metrics)

    summary_table = Table(title="性能评测汇总", caption="基于所有样本的平均值")
    summary_table.add_column("指标", style="cyan")
    summary_table.add_column("值", style="green")
    summary_table.add_column("95%置信区间", style="yellow")

    summary_table.add_row(
        "平均首token响应时间",
        f"{summary['avg_first_token_latency']:.4f} 秒",
        f"±{summary['ftl_95ci']:.4f} 秒"
    )
    summary_table.add_row(
        "平均输出每秒token数",
        f"{summary['avg_tokens_per_second']:.2f} tokens/s",
        f"±{summary['tps_95ci']:.2f} tokens/s"
    )
    summary_table.add_row("样本数量", f"{summary['total_samples']}", "")

    console.print(summary_table)
