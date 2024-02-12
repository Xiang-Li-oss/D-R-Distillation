import argparse
from transformers import T5TokenizerFast, T5ForConditionalGeneration, GPT2Model, GPT2TokenizerFast, GPT2Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from data import DataModule
from Model import Model
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--val_set", type=str, default="dev")
    parser.add_argument("--model", type=str, default="t5-large")
    parser.add_argument("--role", type=str, default="planer")
    parser.add_argument("--val_mode", type=str, default="max")
    parser.add_argument("--val_metric", type=str, default="f1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--devices", type=int, nargs="+", default=9)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default='auto')
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ckpt_version", type=int)
    parser.add_argument("--mode", type=str, default='train',
                        choices=['train', 'test', 'predict'])
    


    args = parser.parse_args()

    if args.precision == 16:
        args.precision = "bf16"
        print("Setting precision to bf16")

    dataset = args.dataset
    
    model_path = args.model
    
    if 'flan' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        tokenizer = T5TokenizerFast.from_pretrained(
            model_path, model_max_length=512)
    

            
    batch_size = args.batch_size

    if args.inference_batch_size is None:
        inference_batch_size = args.batch_size
    else:
        inference_batch_size = args.inference_batch_size

    data_module = DataModule(args, model_path, dataset, tokenizer, batch_size=batch_size, inference_batch_size=inference_batch_size,
                             num_workers=8)
   
    default_root_dir = os.path.join("output_dir", dataset, "{}_{}".format(args.model, args.role))


    lm = Model(args, model, tokenizer)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, mode=args.val_mode, filename='best_{f1:.3f}_{em:.3f}', monitor=args.val_metric)

    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                            default_root_dir=default_root_dir, min_epochs=args.epoch, max_epochs=args.epoch,
                            accumulate_grad_batches=args.accumulate, precision=args.precision,
                            callbacks=[checkpoint_callback])
    trainer.fit(lm, datamodule=data_module)


    
