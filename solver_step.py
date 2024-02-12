from Model import Model
import argparse
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ds_inference_dataloader import build_dataloader_from_list, merge_list, is_finish, build_solver_input
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import json
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="hotpotqa")
parser.add_argument("--model", type=str, default="t5-base")
parser.add_argument("--val_set", type=str, default="dev")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_iters", type=int, default=5)
parser.add_argument("--devices", type=int, nargs="+", default=3)
parser.add_argument("--precision", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--role", type=str, default="planer")
parser.add_argument("--strategy", type=str, default='auto')
parser.add_argument("--role_path", type=str)
parser.add_argument("--file_path", type=str)
args = parser.parse_args()
if args.precision == 16:
    args.precision = "bf16"
    print("Setting precision to bf16")

dataset = args.dataset

model_path = args.model

def solver_step(model_path, datas):
    if 'flan' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        solver_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        solver_model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        tokenizer = T5TokenizerFast.from_pretrained(
            model_path, model_max_length=512)
    solver_kwargs = {
            'args': args,
            'model': solver_model,
            'tokenizer': tokenizer,
            'map_location': lambda storage, loc: storage.cuda(9)
        }
    solver = Model.load_from_checkpoint(args.role_path, **solver_kwargs)
    solver_trainer = pl.Trainer(accelerator='gpu', devices=[9], 
                            strategy=args.strategy, precision=args.precision, enable_checkpointing=False)
    solver_dataloader = build_dataloader_from_list(args, datas, tokenizer)
    solver_outputs = solver_trainer.predict(solver, solver_dataloader)
    solver_outputs = merge_list(solver_outputs)
    return solver_outputs


with open(args.file_path, 'r') as f:
    all_datapoints = json.load(f)

unfinished_datas = [dp for dp in all_datapoints if dp['finish'] == 'False']

solver_datas = []
for dp in tqdm(unfinished_datas):
    solver_datas.append(build_solver_input(dp))

solver_outputs = solver_step(model_path, solver_datas)
assert(len(unfinished_datas) == len(solver_outputs))
for unfinished_data, solver_output in zip(unfinished_datas, solver_outputs):
    solver_output = solver_output.strip(' \n')
    idx = unfinished_data['idx']
    all_datapoints[idx]['input'] = all_datapoints[idx]['input'] + '\nIntermediate answer: ' + solver_output

with open(args.file_path, 'w') as f:
    json.dump(all_datapoints, f, indent=2)