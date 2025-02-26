import time
import torch

from transformers import AutoTokenizer, AutoModel
from glm_config import ProjectConfig


def inference(
        model,
        tokenizer,
        instuction: str,
        sentence: str
):
    """
    模型 inference 函数。

    Args:
        instuction (str): _description_
        sentence (str): _description_

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        input_text = f"Instruction: {instuction}\n"
        if sentence:
            input_text += f"Input: {sentence}\n"
        input_text += f"Answer: "
        batch = tokenizer(input_text, return_tensors="pt")
        out = model.generate(
            input_ids=batch["input_ids"].to(pc.device),
            max_new_tokens=max_new_tokens,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.split('Answer: ')[-1]
        return answer


if __name__ == '__main__':
    from rich import print

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
        print('模型加载模型成功...')
    except Exception as e:
        print(e)
        print('加载模型失败，请检查模型路径是否正确')
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

    print('开始推理...')
    print('=' * 50)
    start = time.time()
    for i, sample in enumerate(samples):
        print(f"人类提问--> {sample['input']}")
        res = inference(
            model,
            tokenizer,
            sample['instruction'],
            sample['input']
        )
        print(f"模型回答 --> {res}")
        print('=' * 50)
    print(f'推理耗时： {round(time.time() - start, 2)}s.')
