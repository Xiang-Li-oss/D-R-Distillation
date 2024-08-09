import os 
import json
from evaluation_util import calculate_metrics
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--result_path", type=str, default='prediction.json')
args = parser.parse_args()

def extract_answer(traj):
    last_line = traj.strip(' \n').split('\n')[-1]
    if ':' in last_line:
        answer = last_line.split(':')[-1]
    else:
        answer = last_line

    return answer
with open(args.result_path, 'r') as f:
    data_points = json.load(f)

preds = []
golds = []

for dp in data_points:
    preds.append(extract_answer(dp['input']))
    golds.append(dp['gold'])


print(calculate_metrics(preds, golds))
