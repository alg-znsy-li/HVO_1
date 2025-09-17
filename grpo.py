import re
import deepspeed
import torch
from datasets import load_dataset
from transformers import (
AutoModelForCausalLM,
AutoModelForSequenceClassification,
AutoTokenizer,
HfArgumentParser,
)
import numpy as np
from trl import ModelConfig
from trl import GRPOConfig,GRPOTrainer
import shutil
import os
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize
from metric.scorer import UniEvaluator
from utils import add_question, print_scores
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json
from dataclasses import dataclass, field
from prettytable import PrettyTable
from utils import add_question, print_scores,convert_to_json
from unieval import UniEvaluator, SumEvaluator
from typing import Optional
from grpotrainer import *
import numpy as np
from random import seed, shuffle
from transformers import set_seed

@dataclass
class ModelArguments:
    unieval_model_name_or_path: str = field(
        default="/lpai/volumes/jfs-ppl-alg-disk-bd-ga/MingZhong--unieval-sum/",  
        metadata={"help": "unieval模型的名称或本地路径"}
    )
    base_model_name_or_path: str = field(
        default="/lpai/volumes/jfs-ppl-alg-disk-bd-ga/songjunjie/ppl_rank/test_model/asllr_model/Qwen1___5-1___8B",  
        metadata={"help": "训练模型的名称或本地路径"}
    )
    unieval_model_deepspeed_config: str = field(
        default='./deepspeed_config.json', 
        metadata={"help": "unieval模型的deepspeed config"}
    )
    dataset_path: str = field(
        default='/lpai/volumes/jfs-ppl-alg-disk-bd-ga/songjunjie/summary_rd/cnn_dailymail/3.0.0', 
        metadata={"help": "数据集地址"}
    )
    random_seed: Optional[int] = field(
        default= 2025, 
        metadata={"help": "随机种子"}
    )
    dataset_sample_train_num: Optional[int] = field(
        default= None, 
        metadata={"help": "数据集train采样量"}
    )
    dataset_sample_eval_num: Optional[int] = field(
        default= None, 
        metadata={"help": "数据集eval采样量"}
    )
    hypervolume_epsilon: Optional[float] = field(
        default= None, 
        metadata={"help": "锚定点的epsilon参数"}
    )
    use_hypervolume: str = field(
        default= 'false', 
        metadata={"help": "是否使用hypervolume方法"}
    )
    mean_compression_ratio: int = field(
        default= 15, 
        metadata={"help": "训练集压缩率"}
    )
    use_concise_reward: str = field(
        default= '1', 
        metadata={"help": "是否使用concise_reward"}
    )
    only_unieval: str = field(
        default= 'false', 
        metadata={"help": "是否只算unieval"}
    )
    concise_steep: int = field(
        default= 4, 
        metadata={"help": "简洁性指数"}
    )
    concise_offset: int = field(
        default= 8, 
        metadata={"help": "简洁性偏移"}
    )
    use_repetition: str = field(
        default= 'true', 
        metadata={"help": "是否使用重复性惩罚"}
    )

    
parser = HfArgumentParser((GRPOConfig, ModelConfig, ModelArguments))
config, model_config, model_args= parser.parse_args_into_dataclasses()
print(f"{'='*50}\nconfig参数内容\n{'='*50}\n{config}")
print(f"{'='*50}\nModelArguments参数内容\n{'='*50}\n{json.dumps(model_args.__dict__, indent=4)}")
shutil.rmtree(config.output_dir, ignore_errors=True)

nproc_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
evaluator = SumEvaluator(max_length=config.max_prompt_length,
                         cache_dir=None, 
                         deepspeed_config = model_args.unieval_model_deepspeed_config,
                         unieval_model_name_or_path= model_args.unieval_model_name_or_path,
                         word_size = nproc_per_node)

def evaluate_reward(completions, origin_text, ref_text, evaluator, dims):
    summarys = [completions[i][0]['content'] for i in range(len(completions))]
    data = convert_to_json(output_list=summarys, src_list=origin_text, ref_list=ref_text)
    eval_scores = evaluator.evaluate(data, dims=dims)
    try:
        print("[summary]\n",summarys[0])
        print("[ref_text]\n",ref_text[0])
        print(eval_scores[0])
    except Exception as e:
        print(str(e))
        pass
    return [item[dims[0]] for item in eval_scores]
    
def coherence_reward(completions, origin_text, ref_text, **kwargs):
    return evaluate_reward(completions, origin_text, ref_text, evaluator, dims=['coherence'])
def consistency_reward(completions, origin_text, ref_text, **kwargs):
    return evaluate_reward(completions, origin_text, ref_text, evaluator, dims=['consistency'])
def fluency_reward(completions, origin_text, ref_text, **kwargs):
    return evaluate_reward(completions, origin_text, ref_text, evaluator, dims=['fluency'])
def relevance_reward(completions, origin_text, ref_text, **kwargs):
    return evaluate_reward(completions, origin_text, ref_text, evaluator, dims=['relevance'])

if  model_args.use_concise_reward.lower() == '1':
    def concise_reward(completions, origin_text, ref_text, **kwargs):
        summarys = [completions[i][0]['content'] for i in range(len(completions))]
        def get_score(summarys,origin_text,mean_compression_ratio,ref_text):
            mean_compression_ratio = mean_compression_ratio
            compression_ratio = len(origin_text)/len(summarys) if len(summarys)>0 else 0
            return 1/(1.1**(abs(compression_ratio - mean_compression_ratio)))
        
        return [get_score(summarys[i],origin_text[i],int(model_args.mean_compression_ratio),ref_text[i]) for i in range(len(summarys))]

if  model_args.use_concise_reward.lower() == '2':
    def concise_reward(completions, origin_text, ref_text, **kwargs):
        summarys = [completions[i][0]['content'] for i in range(len(completions))]
        def get_score(summary,origin_text,mean_compression_ratio,ref_text):
            mean_compression_ratio = mean_compression_ratio
            compression_rate = len(origin_text)/len(summary) if len(summary)>0 else 0
            min_bound = 9
            max_bound = 22
            if compression_rate < min_bound:
                compression_rate = min_bound
            if compression_rate > max_bound:
                compression_rate = max_bound
            return (compression_rate - min_bound)/(max_bound - min_bound )
        return [get_score(summarys[i],origin_text[i],int(model_args.mean_compression_ratio),ref_text[i]) for i in range(len(summarys))]

if  model_args.use_concise_reward.lower() == '3':
    def concise_reward(completions, origin_text, ref_text, **kwargs):
        summarys = [completions[i][0]['content'] for i in range(len(completions))]
        def get_score(summarys,origin_text,mean_compression_ratio,ref_text):
            mean_compression_ratio = mean_compression_ratio
            compression_ratio = len(origin_text)/len(summarys) if len(summarys)>0 else 0
            x = abs(compression_ratio - mean_compression_ratio)
            a = int(model_args.concise_steep)
            b = int(model_args.concise_offset)
            return 1/(1 + (x/b)**a)
        return [get_score(summarys[i],origin_text[i],int(model_args.mean_compression_ratio),ref_text[i]) for i in range(len(summarys))]

def repetition_score(completions, origin_text, ref_text, **kwargs):
        summarys = [completions[i][0]['content'] for i in range(len(completions))]
        def get_word_list(text: str):
            """自动区分中英文，返回词或字列表"""
            if re.search('[\u4e00-\u9fff]', text):
                # 中文逐字
                return list(text)
            else:
                # 英文分词
                return text.split()
            
        def zipngram(text: str, ngram_size: int):
            """
            根据输入文本生成 n 元语法（n-gram）元组。
            """
            words = get_word_list(text)
            return zip(*[words[i:] for i in range(ngram_size)])

        def get_score(completion, origin_text,ngram_size = 2):
            """
            计算文本中重复 n-gram 的惩罚分数
            """
            reward = 0
            if completion == '':
                return 0
            if len(get_word_list(completion)) < ngram_size:
                return 0
                
            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                if ng in ngrams:
                    pass
                ngrams.add(ng)
                total += 1

            scaling = len(ngrams) / total
            reward = scaling
            return reward 

        return [get_score(summarys[i],origin_text[i]) for i in range(len(summarys))]

def build_prompt(text):
    return f"""Text: {text}
               Instruction: Summarize the Text without any Explanation.
               Output:
            """

if 'billsum' in model_args.dataset_path.lower():
    def get_dataset(split="train", num=None, dataset_path= None, random_seed = 2025):
        set_seed(random_seed)
        raw_data =  load_dataset("parquet", data_files = dataset_path +split+"-00000-of-00001.parquet")['train']

        if num is not None:
            raw_data = raw_data.shuffle(seed=random_seed).select(range(num))
        data = raw_data.map(lambda x: {
            'prompt': [
                {'role': 'user', 'content': build_prompt(x["text"])}
            ],
            'origin_text': x["text"],
            'ref_text': x["summary"]
        })
        return data

if 'cnn_dailymail' in model_args.dataset_path.lower():
    def get_dataset(split="train", num=None, dataset_path= None, random_seed = 2025):
        set_seed(random_seed)
        raw_data = load_dataset(dataset_path, split=split)
        if num is not None:
            raw_data = raw_data.shuffle(seed=random_seed).select(range(num))
        data = raw_data.map(lambda x: {
            'prompt': [
                {'role': 'user', 'content': build_prompt(x["article"])}
            ],
            'origin_text': x["article"],
            'ref_text': x["highlights"]
        })
        return data


#### training model ####
# "/lpai/inputs/models/summary-sft-25-06-12-3/"
model_name = model_args.base_model_name_or_path
model = AutoModelForCausalLM.from_pretrained(
    model_name,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("活跃设备：",os.environ.get("WORLD_SIZE", "无"))

# 加载训练数据集合
train_dataset = get_dataset(split = 'train', 
                            num = model_args.dataset_sample_train_num,
                            dataset_path = model_args.dataset_path,
                            random_seed = int(model_args.random_seed)
                           )
eval_dataset =  get_dataset(split = 'test',
                            num = model_args.dataset_sample_eval_num,
                            dataset_path = model_args.dataset_path)


common_params = {
    'model': model,
    'processing_class': tokenizer,
    'reward_funcs': [
        coherence_reward,
        consistency_reward,
        fluency_reward,
        relevance_reward
    ],
    'args': config,
    'train_dataset': train_dataset,
    'eval_dataset': eval_dataset,
    'only_unieval':model_args.only_unieval
}

if model_args.use_concise_reward.lower() != '0':
    print(f"{'='*50}\n使用concise_reward{model_args.use_concise_reward.lower()}\n{'='*50}")
    common_params['reward_funcs'].append(concise_reward)

if model_args.use_repetition.lower() == 'true':
    print(f"{'='*50}\n使用use_repetition\n{'='*50}")
    common_params['reward_funcs'].append(repetition_score)

if model_args.use_hypervolume.lower() == 'true':
    print(f"{'='*50}\n使用hypervolume方法\n{'='*50}")
    common_params['epsilon'] = model_args.hypervolume_epsilon  # 添加 `epsilon` 参数
    trainer = LXGRPOTrainer(**common_params)
else:
    trainer = GRPOTrainerNew(**common_params)
trainer.train()
