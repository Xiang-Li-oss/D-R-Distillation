from torch.utils.data import DataLoader
from typing import List, Dict
from datasets import Dataset
from retrieval import search
import json
from tqdm import tqdm

def list_of_dicts_to_dict_of_lists(list_of_dict):
    dict_of_lists = {}
    for key in list_of_dict[0].keys():
        dict_of_lists[key] = [d[key] for d in list_of_dict]
    return dict_of_lists

def merge_list(list):
    resutlt = []
    for l in list:
        resutlt.extend(l)
    return resutlt

def is_finish(sample):
    input = sample['input']
    input = input.strip(' \n')
    if '\n' not in input:
        last_line = input
    else:
        last_line = input.split('\n')[-1]
    return last_line.startswith('So the final')

def build_reponser_input(dp):
    current_traj = dp['input']
    decomposer_output = current_traj.split('\n')[-1].strip(' \n')
    if not decomposer_output.startswith('Subquestion'):
        print('no subquestion found...' + decomposer_output)
    if ':' not in decomposer_output:
        question = decomposer_output
    else:
        question = decomposer_output.split(':')[-1].strip()
  
    paragraphs= search(question, 3, None)

    responser_input = {
        'input': paragraphs.strip(' \n') + '\n###\n' + question
    }
    return responser_input
    
def build_dataloader_from_list(args, data_points: List[Dict], tokenizer):

    def tokenize(example) -> dict:
        input_max_length = 512
        encoded_inputs = tokenizer(
            example["input"],
            padding="longest",
            max_length=input_max_length,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_inputs["input_ids"]
        attention_mask  = encoded_inputs["attention_mask"]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if "label" in example:
            encoded_outputs = tokenizer(
                example['label'],
                padding="longest",
                max_length=512,
                truncation=True,
                return_tensors='pt'
            )
            label_ids = encoded_outputs["input_ids"]
            decoder_attention_mask = encoded_outputs['attention_mask']
            label_ids[~decoder_attention_mask.bool()] = -100
            result.update({
                "decoder_attention_mask": decoder_attention_mask,
                "labels": label_ids,
            })
        return result

    data_points = list_of_dicts_to_dict_of_lists(data_points)

    dataset= Dataset.from_dict(data_points)
    dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
    dataset.set_format(type='torch', columns=["input_ids", "attention_mask"])

    return DataLoader(
         dataset, 
         batch_size=args.batch_size,
         num_workers=8,
         shuffle=False
    )

