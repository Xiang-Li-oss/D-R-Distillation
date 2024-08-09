from Model import Model
import argparse
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ds_inference_dataloader import build_dataloader_from_list, merge_list, is_finish, build_responser_input
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
parser.add_argument("--role", type=str, default="decomposer")
parser.add_argument("--strategy", type=str, default='auto')
parser.add_argument("--file_path", type=str, default='prediction.json')
args = parser.parse_args()
if args.precision == 16:
    args.precision = "bf16"
    print("Setting precision to bf16")

dataset = args.dataset
args.file_path = '{}-{}-{}'.format(dataset, args.model, args.file_path)


decomposer_path = 'path/to/decomposer_ckpt'
responser_path = 'path/to/responser_ckpt'



model_path = args.model

def decomposer_step(model_path, datas):
    if 'flan' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        decomposer_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        decomposer_model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        tokenizer = T5TokenizerFast.from_pretrained(
            model_path, model_max_length=512)
    decomposer_kwargs = {
            'args': args,
            'model': decomposer_model,
            'tokenizer': tokenizer,
            'map_location': lambda storage, loc: storage.cuda(9)
        }
    decomposer = Model.load_from_checkpoint(decomposer_path, **decomposer_kwargs)
    decomposer_trainer = pl.Trainer(accelerator='gpu', devices=[9], 
                            strategy=args.strategy, precision=args.precision, enable_checkpointing=False)
    decomposer_dataloader = build_dataloader_from_list(args, datas, tokenizer)
    decomposer_outputs = decomposer_trainer.predict(decomposer, decomposer_dataloader)
    decomposer_outputs = merge_list(decomposer_outputs)
    return decomposer_outputs

                            
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


decomposer_outputs = decomposer_step(model_path, all_datapoints)


for idx, decomposer_output in enumerate(decomposer_outputs):
    decomposer_output.strip(' \n')
    if not decomposer_output.startswith('Subquestion') or decomposer_output.startswith('So the final'):
        print(decomposer_output)
    all_datapoints[idx]['input'] = all_datapoints[idx]['input'] + '\n' + decomposer_output

with open(args.file_path, 'w') as f:
    json.dump(all_datapoints, f, indent=2)
        
for i in range(args.max_iters):
    responser_command = 'python responser_step.py --dataset={} --model={} --batch_size={} --role_path={} --file_path={}'.format(
        dataset, args.model, 128, responser_path, args.file_path
    )
    os.system(responser_command)
    time.sleep(1)
    decomposer_command = 'python decomposer_step.py --dataset={} --model={} --batch_size={} --role_path={} --file_path={}'.format(
        dataset, args.model, args.batch_size, decomposer_path, args.file_path
    )
    os.system(decomposer_command)



