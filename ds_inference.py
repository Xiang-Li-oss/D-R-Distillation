from Model import Model
import argparse
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ds_inference_dataloader import build_dataloader_from_list, merge_list, is_finish, build_solver_input
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="hotpotqa")
parser.add_argument("--model", type=str, default="t5-base")
parser.add_argument("--val_set", type=str, default="dev")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_iters", type=int, default=3)
parser.add_argument("--devices", type=int, nargs="+", default=9)
parser.add_argument("--precision", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--role", type=str, default="planer")
parser.add_argument("--strategy", type=str, default='auto')
parser.add_argument("--file_path", type=str, default='prediction.json')
args = parser.parse_args()
if args.precision == 16:
    args.precision = "bf16"
    print("Setting precision to bf16")

dataset = args.dataset
args.file_path = '{}-{}-{}'.format(dataset, args.model, args.file_path)


planer_path = ''
solver_path = ''



model_path = args.model

def planner_step(model_path, datas):
    if 'flan' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        planer_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        planer_model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        tokenizer = T5TokenizerFast.from_pretrained(
            model_path, model_max_length=512)
    planer_kwargs = {
            'args': args,
            'model': planer_model,
            'tokenizer': tokenizer,
            'map_location': lambda storage, loc: storage.cuda(9)
        }
    planer = Model.load_from_checkpoint(planer_path, **planer_kwargs)
    planer_trainer = pl.Trainer(accelerator='gpu', devices=[9], 
                            strategy=args.strategy, precision=args.precision, enable_checkpointing=False)
    planer_dataloader = build_dataloader_from_list(args, datas, tokenizer)
    planer_outputs = planer_trainer.predict(planer, planer_dataloader)
    planer_outputs = merge_list(planer_outputs)
    return planer_outputs

                            
all_datapoints = []
if dataset == 'hotpotqa':
    with open('data/hotpotqa-dev-kilt.jsonl.txt', 'r') as f:
        lines = f.readlines()
        
        for idx, line in enumerate(lines):   
            line = json.loads(line)
            all_datapoints.append({
                'idx': idx,
                'input': 'Question: ' + line['input'],
                'gold': line['output'][0]['answer'],
                'finish': 'False'
            })
elif dataset == 'strategyqa':
    with open('data/strategyqa-dev-all.txt', 'r') as f:
        lines = f.readlines()
        
        for idx, line in enumerate(lines):   
            line = json.loads(line)
            all_datapoints.append({
                'idx': idx,
                'input': 'Question: ' + line['input'],
                'gold': line['output'],
                'finish': 'False'
            })
elif dataset == 'wiki2':
    with open('data/wiki2-dev-all.txt', 'r') as f:
        lines = f.readlines()
        
        for idx, line in enumerate(lines):   
            line = json.loads(line)
            all_datapoints.append({
                'idx': idx,
                'input': 'Question: ' + line['input'],
                'gold': line['output'],
                'finish': 'False'
            })


planer_outputs = planner_step(model_path, all_datapoints)


for idx, planer_output in enumerate(planer_outputs):
    planer_output.strip(' \n')
    if not planer_output.startswith('Subquestion') or planer_output.startswith('So the final'):
        print(planer_output)
    all_datapoints[idx]['input'] = all_datapoints[idx]['input'] + '\n' + planer_output

with open(args.file_path, 'w') as f:
    json.dump(all_datapoints, f, indent=2)
        
for i in range(args.max_iters):
    solver_command = 'python solver_step.py --dataset={} --model={} --batch_size={} --role_path={} --file_path={}'.format(
        dataset, args.model, 128, solver_path, args.file_path
    )
    os.system(solver_command)
    time.sleep(1)
    planer_command = 'python planer_step.py --dataset={} --model={} --batch_size={} --role_path={} --file_path={}'.format(
        dataset, args.model, args.batch_size, planer_path, args.file_path
    )
    os.system(planer_command)



