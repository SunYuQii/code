
import json
import requests
import os
import pickle
from tqdm import tqdm
from setproctitle import setproctitle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    default="fashion", 
                    help="Choose the dataset")
args = parser.parse_args()

dataset = args.dataset
print(dataset)
model = "GLM-4-Flash"
candidate_prompt = json.load(open(f"data/{dataset}/pseudo/candidate_prompt.json", "r"))

def get_response(prompt):
    '''
    '''
    url = ""

    payload = json.dumps({
        "model":model,
        "messages":[{"role":"user",
                    "content":prompt}],
    })
    headers = {
        'Authorization': '',
        'User-Agent': '',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    re_json = json.loads(response.text)

    # print(re_json)
    #{'id': 'chatcmpl-AKhaB7XUfp44KVNGMckYkD8wYOkVF', 'object': 'chat.completion', 'created': 1729497455, 'model': 'gpt-3.5-turbo-0125', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "Based on the user's historical interactions with Williams Lectric Shave products, the user may interact with the following items from the candidate set:\n\n1. Lectric Shave Pre-Shave Original 3 oz\n2. Williams Lectric Shave, 7 Ounce\n\nThese items are related to the user's interest in shaving products and may be of interest to them."}, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 289, 'completion_tokens': 75, 'total_tokens': 364, 'completion_tokens_details': {'reasoning_tokens': 0}, 'prompt_tokens_details': {'cached_tokens': 0}}, 'system_fingerprint': None}
    # print(re_json['choices'][0]['message']['content'])
    return re_json['choices'][0]['message']['content']

llmResRaw = {}
prompt2llmRes = {}
errorIndex = {}
# check whether some response exist in cache
if os.path.exists(f"data/{dataset}/pseudo/llmResRaw.pkl"):
    llmResRaw = pickle.load(open(f"data/{dataset}/pseudo/llmResRaw.pkl", "rb"))

json.dump(llmResRaw, open(f"data/{dataset}/pseudo/llmResRaw.json", "w"))
if os.path.exists(f"data/{dataset}/pseudo/prompt2llmRes.json"):
    prompt2llmRes = json.load(open(f"data/{dataset}/pseudo/prompt2llmRes.json", "r"))
if os.path.exists(f"data/{dataset}/pseudo/errorIndex.json"):
    errorIndex = json.load(open(f"data/{dataset}/pseudo/errorIndex.json","r"))
# avoid broken due to internet connection

thresold = 5
cnt = 0
control = True
tryTimes = 0
while control:
    if len(llmResRaw) == len(candidate_prompt) or len(llmResRaw)+len(errorIndex) >= len(candidate_prompt):
        break
    try:
        # if cnt >= thresold: break
        for key, value in tqdm(candidate_prompt.items()):
            cnt = cnt + 1
            # if cnt >= thresold: break
            if not control:break
            if key not in llmResRaw.keys():
                if value in prompt2llmRes.keys():
                    llmResRaw[key] = prompt2llmRes[value]
                else:
                    if len(value) > 4096:
                        value = value[:4095]
                    try:
                        llmResRaw[key] = get_response(value)
                        prompt2llmRes[value] = llmResRaw[key]
                        print(key,value,llmResRaw[key])
                    except:
                        pickle.dump(llmResRaw, open(f"data/{dataset}/pseudo/llmResRaw.pkl", "wb"))
                        json.dump(prompt2llmRes,open(f"data/{dataset}/pseudo/prompt2llmRes.json", "w"))
                        json.dump(errorIndex,open(f"data/{dataset}/pseudo/errorIndex.json","w"))
                        errorIndex[key] = value
                        continue
    except:
        pickle.dump(llmResRaw, open(f"data/{dataset}/pseudo/llmResRaw.pkl", "wb"))
        json.dump(prompt2llmRes,open(f"data/{dataset}/pseudo/prompt2llmRes.json", "w"))
        json.dump(errorIndex,open(f"data/{dataset}/pseudo/errorIndex.json","w"))

pickle.dump(llmResRaw, open(f"data/{dataset}/pseudo/llmResRaw.pkl", "wb"))
json.dump(prompt2llmRes,open(f"data/{dataset}/pseudo/prompt2llmRes.json", "w"))
json.dump(errorIndex,open(f"data/{dataset}/pseudo/errorIndex.json","w"))