# -*- coding:utf-8 -*-
import torch


class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.pre_model = '/root/autodl-fs/chatglm-6b'
        self.train_path = '/root/autodl-tmp/lora_chatglm/data/mixed_train_dataset.jsonl'
        self.dev_path = '/root/autodl-tmp/lora_chatglm/data/mixed_train_dataset.jsonl'
        self.use_lora = True
        self.use_ptuning = False

        # 调参啦
        self.lora_rank = 4
        self.lora_alpha = 8
        self.batch_size = 1
        self.epochs = 30
        self.learning_rate = 3e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.06

        self.max_source_seq_len = 400
        self.max_target_seq_len = 300
        self.logging_steps = 100
        self.save_freq = 1
        self.pre_seq_len = 128
        self.prefix_projection = False  # 默认为False,即p-tuning,如果为True，即p-tuning-v2
        self.save_dir = '/root/autodl-fs/trained_models/chatglm_6b'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(f"模型保存路径 --> {pc.save_dir}")
    print(f"训练设备 --> {pc.device}")
