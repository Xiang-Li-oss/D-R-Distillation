from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import PreTrainedTokenizer

import torch
import os
import json
from evaluation_util import calculate_metrics


class Model(pl.LightningModule):
    def __init__(self, args, model, tokenizer: PreTrainedTokenizer, use_cpu_offload=False,
                truncate_early=True, max_length=32):
        """
        - completion_metadata: metaddata used to save completions. If None, completions are not saved.
          `epoch_N` is appended to the `train_key` when saving intermediate validation completions.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.tokenizer = tokenizer
        self.use_cpu_offload = use_cpu_offload
        self.lr = args.lr
        self.max_length = max_length
        self.truncate_early = truncate_early
        self.outputs = []
        self.args = args
     
        self.gold = []
        with open('data/{}-{}-{}.txt'.format(self.args.dataset, self.args.val_set, self.args.role), 'r') as f:
            lines = f.readlines()
            self.gold = [json.loads(line)['output'] for line in lines]

    def training_step(self, batch, batch_idx):
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        
        kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        loss = self.model(**kwargs)["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

        kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        loss = self.model(**kwargs)["loss"]
        self.log("val_loss", loss,
                 on_epoch=True, prog_bar=True)
      
        pred = self.model.generate(batch["input_ids"], max_length=self.max_length) #[B, S]
       
        output = {
            "pred": pred
        }
        self.outputs.append(output)
        return output
    
    def on_validation_epoch_end(self) -> None:
       
        pred_strs = []
        for batch_output in self.outputs:
            batch_decode = self.tokenizer.batch_decode(batch_output['pred'], skip_special_tokens=True)
            pred_strs.extend(batch_decode)

        
      
        custom_metrics = calculate_metrics(pred_strs, self.gold)
        self.log("em", custom_metrics['em'], on_epoch=True, prog_bar=True)
        self.log("f1", custom_metrics['f1'], on_epoch=True, prog_bar=True)
        self.outputs.clear()


       
    def predict_step(self, batch, batch_idx):
        pred = self.model.generate(batch['input_ids'], max_length=self.max_length)
        decoded_output = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        return decoded_output

    
    

    def configure_optimizers(self):
        # if self.use_cpu_offload:
        #     optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        # else:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer