from collections import defaultdict
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    default="fashion", 
                    help="Choose the dataset")
args = parser.parse_args()
dataset = args.dataset

raw_list = json.load(open(f"data/{dataset}/pseudo/raw_list.json", "r"))
cand_list = json.load(open(f"data/{dataset}/pseudo/cand_list.json", "r"))

cand2id = json.load(open(f"data/{dataset}/pseudo/candT2id.json", "r"))

id_map = json.load(open(f"data/{dataset}/handled/id_map.json", "r"))

item_dict = json.load(open(f"data/{dataset}/handled/item2attributes.json", "r"))
def itemMap(LLMResp,pool):
        '''

        '''

        vectorizer = TfidfVectorizer()

        all_items = LLMResp + pool

        tfidf_matrix = vectorizer.fit_transform(all_items)

        similarity_matrix = cosine_similarity(tfidf_matrix)

        MapRes = []

        for i, target_item in enumerate(LLMResp):

            similarities = similarity_matrix[i, len(LLMResp):]

            most_similar_index = np.argsort(similarities)[::-1][0]

            MapRes.append(pool[most_similar_index])
        return MapRes

if __name__ == "__main__":
    candidate_data = {}
    title2id = {}
    notitle = []
    pseudoitem_data = {}

    noid = []

    for user in raw_list.keys():

        # candidate_data[user] = itemMap(raw_list[user],cand_list[user])
        for elem in raw_list[user]:
            try:
                if user not in pseudoitem_data.keys():
                    pseudoitem_data[user] = [cand2id[user][elem]]
                    # pseudoitem_data[user] = [title2id[elem.strip()]]
                else:
                    pseudoitem_data[user].append(cand2id[user][elem])
                    # pseudoitem_data[user].append(title2id[elem.strip()])
            except:
                print(f"error at {elem}")
        

    # print(len(noid))

    User = defaultdict(list)    # default value is a blank list
    f = open(f'./data/{dataset}/handled/inter.txt', 'r')
    for line in f:  # use a dict to save all seqeuces of each user
        u, i = line.rstrip().split(' ')
        u = str(u)
        i = int(i)
        User[u].append(i)

    aug_data = {}
    for user in pseudoitem_data.keys():
        aug_data[user] = pseudoitem_data[user] + User[user]

    with open(f"data/{dataset}/handled/interAug.txt", 'w') as f:
        for user, item_list in tqdm(aug_data.items()):
            for item in item_list:
                u = int(user)
                i = int(item)
                f.write('%s %s\n' % (u, i))
