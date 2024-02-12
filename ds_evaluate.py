import os 
import json
from evaluation_util import calculate_metrics
def extract_answer(traj):
    last_line = traj.strip(' \n').split('\n')[-1]
    if ':' in last_line:
        answer = last_line.split(':')[-1]
    else:
        answer = last_line

    return answer
with open('hotpotqa-t5-base-prediction.json', 'r') as f:
    data_points = json.load(f)

preds = []
golds = []

for dp in data_points:
    preds.append(extract_answer(dp['input']))
    golds.append(dp['gold'])


print(calculate_metrics(preds, golds))
