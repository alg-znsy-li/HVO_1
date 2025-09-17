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
from trl import GRPOConfig, GRPOTrainer
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

class UniEvaluator:
    def __init__(self, model_name_or_path, max_length=1024, device='cuda:0', cache_dir=None, deepspeed_config=None, word_size=None):
        """ Set up model """
        self.device = device
        self.max_length = max_length
        self.deepspeed_config = deepspeed_config 
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config,
                                                           cache_dir=cache_dir)
        self.model.eval()
        if deepspeed_config:
            with open(deepspeed_config, 'r') as f:
                self.deepspeed_config = json.load(f)
                self.deepspeed_config['tensor_parallel']['tp_size'] = word_size
            self.model = deepspeed.init_inference(self.model, config=self.deepspeed_config)
        else:
            self.model.to(device)
        self.softmax = nn.Softmax(dim=1)

        self.pos_id = self.tokenizer("Yes")["input_ids"][0]
        self.neg_id = self.tokenizer("No")["input_ids"][0]

    def score(self, inputs, batch_size=8):
        """
            Get scores for the given samples.
            final_score = postive_score / (postive_score + negative_score)
        """

        # The implementation of "forward" in T5 still requires decoder_input_ids.
        # Therefore, we construct a random one-word target sequence.
        # The content of the target has no effect on the final scores.
        tgts = ["No" for _ in range(len(inputs))]

        pos_score_list, neg_score_list = [], []
        for i in tqdm(range(0, len(inputs), batch_size)):
            src_list = inputs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    device = self.model.module.device
                    src_tokens = encoded_src['input_ids'].to(device)
                    src_mask = encoded_src['attention_mask'].to(device)
                    
                    tgt_tokens = encoded_tgt['input_ids'].to(device)[:, 0].unsqueeze(-1)
                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels = tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
            
                    pos_score = self.softmax(logits)[:, self.pos_id] # Yes
                    neg_score = self.softmax(logits)[:, self.neg_id] # No

                    cur_pos_score = [x.item() for x in pos_score]
                    cur_neg_score = [x.item() for x in neg_score]
                    pos_score_list += cur_pos_score
                    neg_score_list += cur_neg_score

            except RuntimeError:
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        
        score_list = []
        for i in range(len(pos_score_list)):
            score_list.append(pos_score_list[i] / (pos_score_list[i] + neg_score_list[i]))
            
        return score_list

class SumEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None, deepspeed_config =None, unieval_model_name_or_path = None, word_size = None):
        """ Set up evaluator for text summarization """
        self.scorer = UniEvaluator(model_name_or_path=unieval_model_name_or_path, 
                                   max_length=max_length, 
                                   device=device, 
                                   cache_dir=cache_dir,
                                   deepspeed_config = deepspeed_config,
                                   word_size = word_size
                                  )
        self.task = 'summarization'
        self.dimensions = ['coherence', 'consistency', 'fluency', 'relevance']
    
    def evaluate(self, data, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, SumEvaluator will evaluate
                  four dimensions: coherence, consistency, fluency, relevance.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.
                     
            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            # Calculate average sentence-level scores for 'consistency' and 'fluency'
            if dim == 'consistency' or dim == 'fluency':
                src_list, output_list = [], []
                n_sents = [] # the number of sentences in each generated summary
                for i in range(n_data):
                    if dim == 'consistency':
                        source = data[i]['source']
                    else:
                        source = ''
                    system_outputs = sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(system_outputs))
                    for j in range(len(system_outputs)):
                        src_list.append(source)
                        output_list.append(system_outputs[j])
                input_list = add_question(dimension=dim, output=output_list, 
                                          src=src_list, task=self.task)
                sent_score = self.scorer.score(input_list)
                
                # Get average score for each sample
                start_idx = 0
                score = []
                for cur_n_sent in n_sents:
                    try:
                        score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]) / cur_n_sent)
                    except:
                        score.append(0)
                    start_idx += cur_n_sent
            
            # Calculate summary-level score for 'coherence' and 'relevance'
            elif dim == 'coherence' or dim == 'relevance':
                src_list, output_list, ref_list = [], [], []
                for i in range(n_data):
                    src_list.append(data[i]['source'])
                    output_list.append(data[i]['system_output'])
                    if dim == 'relevance':
                        ref_list.append(data[i]['reference'])
                input_list = add_question(dimension=dim, output=output_list, 
                                          src=src_list, ref=ref_list, task=self.task)
                score = self.scorer.score(input_list)
            
            # Please customize other dimensions here for summarization
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')
            
            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores