'''
'''

import random
import numpy as np
import torch
from tqdm import tqdm
import os
import pandas as pd

def set_seed(seed):
    '''
    Fix all of random seed for reproducible training
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # only add when conv in your model

def get_n_params(model):
    '''
    Get the number of parameters of model
    '''
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def unzip_data(data):
    '''

    '''
    res = []
    for user in tqdm(data):

        user_seq = data[user]
        res.append(user_seq)

    return res

def random_neq(l, r, s=[]):
    '''

    '''
    
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def random_select(all_item, s=[]):
    '''
    '''
    t = random.choice(all_item)
    while t in s:
        t = random.choice(all_item)
    return t

def metric_report(data_rank, topk=10):
    '''

    '''

    NDCG, HT = 0, 0
    
    for rank in data_rank:

        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return {'NDCG@10': NDCG / len(data_rank),
            'HR@10': HT / len(data_rank)}

def concat_data(data_list):
    '''

    '''
    res = []


    if len(data_list) == 2:

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            res.append(train[user]+valid[user])
    

    elif len(data_list) == 3:

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:

            res.append(train[user]+valid[user]+test[user])

    else:

        raise ValueError

    return res

def metric_report(data_rank, topk=10):
    '''

    '''

    NDCG, HT = 0, 0
    
    for rank in data_rank:

        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return {f'NDCG@{topk}': NDCG / len(data_rank),
            f'HR@{topk}': HT / len(data_rank)}

def metric_len_report(data_rank, data_len, topk=10, aug_len=0, args=None):
    '''

    '''
    if args is not None:
        ts_user = args.ts_user
    else:
        ts_user = 10

    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0
    count_s = len(data_len[data_len<ts_user+aug_len])
    count_l = len(data_len[data_len>=ts_user+aug_len])

    for i, rank in enumerate(data_rank):

        if rank < topk:

            if data_len[i] < ts_user+aug_len:
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            else:
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1

    return {'Short NDCG@10': NDCG_s / count_s if count_s!=0 else 0, # avoid division of 0
            'Short HR@10': HT_s / count_s if count_s!=0 else 0,
            'Long NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
            'Long HR@10': HT_l / count_l if count_l!=0 else 0,}

def metric_pop_report(data_rank, pop_dict, target_items, topk=10, aug_pop=0, args=None):
    """
    Report the metrics according to target item's popularity
    item_pop: the array of the target item's popularity
    """
    if args is not None:
        ts_tail = args.ts_item
    else:
        ts_tail = 20

    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0
    item_pop = pop_dict[target_items.astype("int64")]
    count_s = len(item_pop[item_pop<ts_tail+aug_pop])
    count_l = len(item_pop[item_pop>=ts_tail+aug_pop])

    for i, rank in enumerate(data_rank):

        if i == 0:  # skip the padding index
            continue

        if rank < topk:

            if item_pop[i] < ts_tail+aug_pop:
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            else:
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1

    return {'Tail NDCG@10': NDCG_s / count_s if count_s!=0 else 0,
            'Tail HR@10': HT_s / count_s if count_s!=0 else 0,
            'Popular NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
            'Popular HR@10': HT_l / count_l if count_l!=0 else 0,}

def metric_len_5group(pred_rank, 
                      seq_len, 
                      thresholds=[5, 10, 15, 20], 
                      topk=10):
    '''
    '''
    NDCG = np.zeros(5)
    HR = np.zeros(5)    
    for i, rank in enumerate(pred_rank):

        target_len = seq_len[i]
        if rank < topk:

            if target_len < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_len < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            elif target_len < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

            elif target_len < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1

            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    count[0] = len(seq_len[seq_len>=0]) - len(seq_len[seq_len>=thresholds[0]])
    count[1] = len(seq_len[seq_len>=thresholds[0]]) - len(seq_len[seq_len>=thresholds[1]])
    count[2] = len(seq_len[seq_len>=thresholds[1]]) - len(seq_len[seq_len>=thresholds[2]])
    count[3] = len(seq_len[seq_len>=thresholds[2]]) - len(seq_len[seq_len>=thresholds[3]])
    count[4] = len(seq_len[seq_len>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count

def metric_len_3group(pred_rank, 
                      seq_len, 
                      thresholds=[5, 10], 
                      topk=10):
    '''
    '''
    NDCG = np.zeros(3)
    HR = np.zeros(3)    
    for i, rank in enumerate(pred_rank):

        target_len = seq_len[i]
        if rank < topk:

            if target_len < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_len < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            else:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

    count = np.zeros(3)
    count[0] = len(seq_len[seq_len>=0]) - len(seq_len[seq_len>=thresholds[0]])
    count[1] = len(seq_len[seq_len>=thresholds[0]]) - len(seq_len[seq_len>=thresholds[1]])
    count[2] = len(seq_len[seq_len>=thresholds[1]])

    for j in range(3):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count

def metric_len_4group(pred_rank, 
                      seq_len, 
                      thresholds=[5, 10, 15], 
                      topk=10):
    '''

    '''
    NDCG = np.zeros(4)
    HR = np.zeros(4)    
    for i, rank in enumerate(pred_rank):

        target_len = seq_len[i]
        if rank < topk:

            if target_len < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_len < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            elif target_len < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

            else:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1

    count = np.zeros(4)
    count[0] = len(seq_len[seq_len>=0]) - len(seq_len[seq_len>=thresholds[0]])
    count[1] = len(seq_len[seq_len>=thresholds[0]]) - len(seq_len[seq_len>=thresholds[1]])
    count[2] = len(seq_len[seq_len>=thresholds[1]]) - len(seq_len[seq_len>=thresholds[2]])
    count[3] = len(seq_len[seq_len>=thresholds[2]])

    for j in range(4):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count

def metric_pop_5group(pred_rank, 
                      pop_dict, 
                      target_items, 
                      thresholds=[10, 30, 60, 100], 
                      topk=10):
    '''
    '''
    NDCG = np.zeros(5)
    HR = np.zeros(5)    
    for i, rank in enumerate(pred_rank):

        target_pop = pop_dict[int(target_items[i])]
        if rank < topk:

            if target_pop < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_pop < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            elif target_pop < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

            elif target_pop < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1

            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    pop = pop_dict[target_items.astype("int64")]
    count[0] = len(pop[pop>=0]) - len(pop[pop>=thresholds[0]])
    count[1] = len(pop[pop>=thresholds[0]]) - len(pop[pop>=thresholds[1]])
    count[2] = len(pop[pop>=thresholds[1]]) - len(pop[pop>=thresholds[2]])
    count[3] = len(pop[pop>=thresholds[2]]) - len(pop[pop>=thresholds[3]])
    count[4] = len(pop[pop>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count

def record_csv(args, res_dict, path='log'):
    '''
    '''
    
    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    # model_name = args.aug_file + '-' + args.now_str
    model_name = args.now_str
    columns = list(res_dict.keys())
    columns.insert(0, "model_name")
    res_dict["model_name"] = model_name
    # columns = ["model_name", "HR@10", "NDCG@10", "Short HR@10", "Short NDCG@10", "Medium HR@10", "Medium NDCG@10", "Long HR@10", "Long NDCG@10",]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):

        df = pd.DataFrame(new_res_dict)
        df = df[columns]    # reindex the columns
        df.to_csv(csv_path, index=False)

    else:

        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)