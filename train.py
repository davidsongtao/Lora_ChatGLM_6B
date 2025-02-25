import gc
import os
import time
import copy
import argparse
from functools import partial
import peft
import torch.nn.utils
# autocast是PyTorch中一种混合精度的技术，可在保持数值精度的情况下提高训练速度和减少显存占用。
# 该方法混合精度训练，如果在CPU环境中不起任何作用
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_scheduler, AutoModelForCausalLM
from utils.common_utils import *
from data_handle.data_loader import *
from glm_config import *

pc = ProjectConfig()


def evaluate_model(model, dev_dataloader):
    """
    改进的模型评估函数，增加了困惑度(perplexity)计算
    """
    print("该轮次训练结束。开始模型评估...")
    model.eval()
    loss_list = []
    total_correct = 0
    total_tokens = 0
    all_losses = []  # 用于计算困惑度

    with torch.no_grad():
        for batch in dev_dataloader:
            input_ids = batch['input_ids'].to(dtype=torch.long, device=pc.device)
            labels = batch['labels'].to(dtype=torch.long, device=pc.device)

            # 获取模型输出
            if pc.use_lora:
                with autocast():
                    outputs = model(input_ids=input_ids, labels=labels)
            else:
                outputs = model(input_ids=input_ids, labels=labels)

            loss = outputs.loss
            loss_list.append(float(loss.cpu().detach()))

            # 记录每个token位置的loss，用于计算困惑度
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))
            mask = shift_labels.view(-1) != -100
            all_losses.extend(token_loss[mask].cpu().tolist())

            # 计算准确率
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100

            # 准确率计算也使用shift版本保持一致
            shift_predictions = torch.argmax(shift_logits, dim=-1).view(-1)
            shift_mask = shift_labels.view(-1) != -100
            correct = (shift_predictions == shift_labels.view(-1)) & shift_mask

            total_correct += correct.sum().item()
            total_tokens += shift_mask.sum().item()

    # 计算平均损失、困惑度和准确率
    avg_loss = sum(loss_list) / len(loss_list)
    perplexity = torch.exp(torch.tensor(sum(all_losses) / len(all_losses))).item()
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    model.train()
    return avg_loss, accuracy, perplexity


def model2train():
    # 加载分词器，加载模型配置
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True, revision="main")
    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True, revision="main")
    global_step, best_eval_loss, best_epoch = 0, float('inf'), 0

    # 如果使用p-tuning,则在模型中添加一个前缀编码器。
    if pc.use_ptuning:
        config.pre_seq_len = pc.pre_seq_len
        config.prefix_projection = pc.prefix_projection

    # 加载预训练模型
    model = AutoModel.from_pretrained(pc.pre_model,
                                      config=config,
                                      trust_remote_code=True,
                                      revision="main")

    # model.half()将模型数据类型从默认的float32精度转换为更低的float16精度，减少内存
    # model = model.float()
    # print(f"模型 --> {model}")

    # 梯度检查点是一种优化技术，用于在反向传播过程中降低内存使用
    # 保存部分激活值，未保存的反向传播时重新计算
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 不进行缓存，减少内存
    model.config.use_cache = False

    if pc.use_ptuning:
        model.transformer.prefix_encoder.float()
    # print(f'model.lm_head-->{model.lm_head}')
    if pc.use_lora:
        model.lm_head = CastOutputToFloat(model.lm_head)

        # 配置lora超参数
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,  # 推理时为True，比如绝定是否使用dropout
            r=pc.lora_rank,  # 低秩矩阵维度
            lora_alpha=pc.lora_alpha,  # 缩放系数，可以看做是Lora部分的学习率缩放，与全局学习率协同工作。
            lora_dropout=0.1,
        )
        model = peft.get_peft_model(model, peft_config)

    # print(f'model2-->{model}')
    model = model.to(pc.device)
    print(f'模型训练参数:', )
    model.print_trainable_parameters()

    # 配置优化器不需要权重衰减的部分
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)

    # 加载训练集和验证集
    train_dataloader, dev_dataloader = get_data()

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)

    # 指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = pc.epochs * num_update_steps_per_epoch

    # 学习率预热阶段的训练步数
    warm_steps = int(pc.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    # 初始化损失列表，用于记录训练过程中损失值的变化
    loss_list = []
    tic_train = time.time()

    # 开始训练轮次
    for epoch in range(1, pc.epochs + 1):
        print("开始训练...")
        # 为每个logging间隔重置统计数据
        total_correct = 0
        total_tokens = 0
        loss_list = []

        # 开始训练
        for i, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(dtype=torch.long, device=pc.device)
            labels = batch['labels'].to(dtype=torch.long, device=pc.device)

            if pc.use_lora:
                with autocast():
                    outputs = model(input_ids=input_ids, labels=labels)
            else:
                outputs = model(input_ids=input_ids, labels=labels)

            # 保存原始损失用于记录 - 这是关键修改点
            original_loss = outputs.loss
            loss_list.append(float(original_loss.cpu().detach()))  # 在缩放前记录损失

            # 计算训练时的准确率
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_predictions = torch.argmax(shift_logits, dim=-1).view(-1)
            shift_mask = shift_labels.view(-1) != -100
            correct = (shift_predictions == shift_labels.view(-1)) & shift_mask

            total_correct += correct.sum().item()
            total_tokens += shift_mask.sum().item()

            # 为反向传播准备可能缩放的损失
            if pc.grad_accumulation_steps > 1:
                scaled_loss = original_loss / pc.grad_accumulation_steps
            else:
                scaled_loss = original_loss

            scaled_loss.backward()  # 使用可能缩放的损失进行反向传播

            torch.nn.utils.clip_grad_norm_(model.parameters(), pc.max_grad_norm)

            if (i + 1) % pc.grad_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % pc.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                train_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

                print("第 %d 步( %02.2f%% ) | 第 %d 轮 | 损失值: %.5f | 准确率: %.2f%% | 速度: %.2f 步/秒 | 预计耗时: %s"
                      % (
                          global_step,
                          global_step / max_train_steps * 100,
                          epoch,
                          loss_avg,
                          train_accuracy * 100,
                          pc.logging_steps / time_diff,
                          second2time(int(max_train_steps - global_step) / (pc.logging_steps / time_diff))
                      ))
                tic_train = time.time()

        # 评估模型
        average_loss, eval_accuracy, perplexity = evaluate_model(model, dev_dataloader)

        print(f"第{epoch}轮次训练结束。平均损失值: {average_loss:.5f} | 准确率: {(eval_accuracy * 100):.2f}% | 困惑度： {perplexity:.2f} ")

        if average_loss < best_eval_loss and epoch >= 10:
            print(
                f"最小验证损失值已更新: {best_eval_loss:.5f} --> {average_loss:.5f}"
            )
            best_eval_loss = average_loss
            best_epoch = epoch
            best_save_dir = os.path.join(pc.save_dir, "model_best")
            save_model(model, best_save_dir)
            tokenizer.save_pretrained(best_save_dir)
            print(f'最佳模型已保存至 {best_save_dir}...')
        tic_train = time.time()
    print(f"训练结束，最佳模型来自第{best_epoch}轮，验证损失为{best_eval_loss:.5f}")


if __name__ == '__main__':
    model2train()
