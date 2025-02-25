import torch
import torch.nn as nn
from glm_config import *
import copy

pc = ProjectConfig()


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def second2time(seconds: int):
    """
    将秒转换成时分秒。

    Args:
        seconds (int): _description_
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


# def save_model(
#         model,
#         cur_save_dir: str
# ):
#     """
#     存储当前模型。
#     Args:
#         cur_save_path (str): 存储路径。
#     """
#     if pc.use_lora:  # merge lora params with origin model
#         merged_model = copy.deepcopy(model)
#         # 如果直接保存，只保存的是adapter也就是lora模型的参数
#         merged_model = merged_model.merge_and_unload()
#         merged_model.save_pretrained(cur_save_dir)
#     else:
#         model.save_pretrained(cur_save_dir)


def save_model(model, cur_save_dir: str):
    """
    存储当前模型。V2：由于第二个轮次L20显卡爆显存。将模型拉入CPU保存，再拉回GPU
    Args:
        model: 要保存的模型
        cur_save_dir (str): 存储路径。
    """
    # 记录原始设备
    original_device = next(model.parameters()).device

    # 将模型移动到CPU
    model = model.cpu()

    if pc.use_lora:  # merge lora params with origin model
        merged_model = copy.deepcopy(model)
        # 如果直接保存，只保存的是adapter也就是lora模型的参数
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(cur_save_dir)
    else:
        model.save_pretrained(cur_save_dir)

    # 清理缓存
    torch.cuda.empty_cache()

    # 将模型移回原始设备
    model = model.to(original_device)
