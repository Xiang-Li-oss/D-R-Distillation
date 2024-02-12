from typing import List, Dict
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader

from datasets import Dataset
import copy
import sys
import json
from transformers import AutoConfig, AutoTokenizer


def list_of_dicts_to_dict_of_lists(list_of_dict):
    dict_of_lists = {}
    for key in list_of_dict[0].keys():
        dict_of_lists[key] = [d[key] for d in list_of_dict]
    return dict_of_lists

class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            args,
            model_name,
            dataset_name,
            tokenizer,
            batch_size,
            inference_batch_size,
            num_workers,
            **kwargs
            ) -> None:
        super().__init__()

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.role = args.role
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.args = args


    def setup(self, stage:str):
        train_datapoints = self.load_data(split='train')
        
        test_datapoints = self.load_data(split=self.args.val_set)
      
        train_datapoints, test_datapoints = \
            self.format_samples(train_datapoints, include_labels=True), \
            self.format_samples(test_datapoints, include_labels=True)
        
        train_datapoints = list_of_dicts_to_dict_of_lists(train_datapoints)
        test_datapoints = list_of_dicts_to_dict_of_lists(test_datapoints)
      
        
        train_dataset = Dataset.from_dict(train_datapoints)
        test_dataset = Dataset.from_dict(test_datapoints)

        self.train_dataset = train_dataset.map(self.tokenize, batched=True, batch_size=len(train_dataset))
        self.test_dataset = test_dataset.map(self.tokenize, batched=True, batch_size=len(test_dataset))
       

        self.train_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"])
        self.test_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"])
    

    def tokenize(self, example) -> dict:
        input_max_length = 512
        encoded_inputs = self.tokenizer(
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
            encoded_outputs = self.tokenizer(
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

  
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        
            
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
         return DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    # def predict_dataloader(self) -> EVAL_DATALOADERS:
    #     if self.predict_split == 'train':
    #         return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.inference_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False
    #     ) 
    #     elif self.predict_split == 'test':
    #         return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.inference_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False
    #     ) 
    
    def load_data(self, split):
       
        if split == 'train':
            path = 'data/{}-train-{}.txt'.format(self.dataset_name, self.role)
        elif split in ['test', 'dev']:
            path = 'data/{}-{}-{}.txt'.format(self.dataset_name, self.args.val_set, self.role)
            
        with open(path, 'r') as f:
            lines  = f.readlines()
            datas = [json.loads(line) for line in lines]
        return datas
    
    

    def format_samples(self, samples: List[Dict], include_labels: bool):
        formatted_samples = []

        errors = 0
        for sample in samples:
            try:
                formatted_sample = self.format_sample(sample, include_labels)
            except ValueError as e:
                errors += 1
                continue
            formatted_samples.append(formatted_sample)

        if errors > 0:
            print("ERROR: {}/{} samples could not be formatted".format(errors, len(samples)), file=sys.stderr)
            print("Raising last Exception", file=sys.stderr)
            raise e
        
        return formatted_samples


    def format_sample(self, sample: Dict, include_label: bool):
        sample = copy.deepcopy(sample)

        
        result = {}
        if self.role == 'planer':
            result['input'] = 'Question: ' + sample['input']
        else:
            result['input'] = sample['input']

        if include_label:
           result['label'] = sample['output']
     


        return result