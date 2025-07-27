'''
input file: llmResRaw.pkl cand_list.json
output file: raw_list.json
'''

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
from tqdm import tqdm
from setproctitle import setproctitle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    default="fashion", 
                    help="Choose the dataset")
parser.add_argument("--gpu_id",
                    default=0,
                    help="the gpu id")
args = parser.parse_args()

dataset = args.dataset
print(dataset)

num2match = 1

gpu_id = args.gpu_id


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
model.to(device)


# error_list = json.load(open(f"data/{dataset}/reliability/error_list.json", "r"))

llmResRaw = pickle.load(open(f"data/{dataset}/reliability/llmResRaw.pkl", "rb"))

cand_list = json.load(open(f"data/{dataset}/reliability/cand_list.json", "r"))

# raw_list = json.load(open(f"data/{dataset}/reliability/raw_list.json", "r"))

def encode_text(text):
    """  """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def get_item_match(sentence, item_set):
    """

    """

    sentence_vector = encode_text(sentence)


    item_vectors = {item: encode_text(item) for item in item_set}


    similarities = {item: cosine_similarity([sentence_vector], [vector])[0][0] for item, vector in item_vectors.items()}


    sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items[:num2match]  


result = {}
user2pseu = {}

for user in tqdm(llmResRaw.keys()):
    result[user] = {}
    sentence = llmResRaw[str(user)]
    item_set = cand_list[str(user)]

    matched_items = get_item_match(sentence, item_set)
    # result[user]['rawPrompt'] = sentence
    result[user]['matchItems'] = [elem[0] for elem in matched_items]
    user2pseu[user] = result[user]['matchItems']

# json.dump(result, open(f"data/{dataset}/reliability/humanFix.json", "w"))
json.dump(user2pseu,open(f"data/{dataset}/reliability/raw_list.json","w"))