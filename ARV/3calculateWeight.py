'''
input file: userMask.json,raw_list.json,interAug.txt
output file: weight.txt
'''

from collections import defaultdict
import json
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from setproctitle import setproctitle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    default="fashion", 
                    help="Choose the dataset")
parser.add_argument("--pseNum",
                    default=3,
                    help="the number of the pse num")
parser.add_argument("--gpu_id",
                    default=0,
                    help="the gpu id")
args = parser.parse_args()

dataset = args.dataset
print(dataset)
pseNum = args.pseNum

# decayThreshold = 5
# decayRate = 0.5

maskedLabel = json.load(open(f"data/{dataset}/reliability/userMask.json", "r"))
userLabel = maskedLabel['userTitle']

llmPre = json.load(open(f"data/{dataset}/reliability/raw_list.json", "r"))

user2Weight = {}
gpu_id = args.gpu_id


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
model.to(device)

def encode_text(text):
    """"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

for user in tqdm(llmPre.keys()):

    llmPre_vector = encode_text(llmPre[user])

    label_vector = encode_text(userLabel[user])

    similarity = cosine_similarity([llmPre_vector], [label_vector])[0][0]
    user2Weight[user] = similarity
    

with open(f"data/{dataset}/handled/weight{pseNum}.txt", 'w') as f:
    for user in tqdm(user2Weight.keys()):
        u = int(user)
        weight = round(user2Weight[user],6)
        f.write('%s %s\n' % (u, weight))