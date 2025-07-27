from collections import defaultdict
from torch.utils.data import Dataset
import copy
import numpy as np
from utils.utils import random_neq,random_select

class Seq2SeqDataset(Dataset):
    '''
    The train dataset for Sequential recommendation with seq-to-seq loss
    '''

    def __init__(self, data, item_num, max_len, neg_num,all_items):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions"]
        self.all_items = all_items

    def __len__(self):
        '''
        '''
        return len(self.data)
    
    def __getitem__(self, index):
        '''
        '''

        inter = self.data[index]

        non_neg = copy.deepcopy(inter)
        

        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)

        nxt = inter[-1]

        idx = self.max_len - 1
        
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)

            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:

            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions
    
class SeqDataset(Dataset):
    '''
    The train dataset for Sequential recommendation
    '''

    def __init__(self, data, item_num, max_len, neg_num,all_items):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions"]
        self.all_items = all_items


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            # per_neg = random_select(self.all_items,non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions
    
class Seq2SeqDatasetAllUser(Seq2SeqDataset):
    '''
    '''
    def __init__(self, data, item_num, max_len, all_items, neg_num):

        super().__init__(data, item_num, max_len, neg_num,all_items)
        self.var_name = ["seq", "pos", "neg", "positions", "user_id"]
        self.all_items = all_items
    
    def __getitem__(self, index):
        '''

        '''

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []

        neg = np.array(self.all_items)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:

            mask_len = self.max_len - (len(inter))

            positions = list(range(1, len(inter)+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        
        return seq, pos, neg, positions ,index
    
class AugSeqDataset(Dataset):
    '''
    The train dataset for Sequential recommendation with seq-to-seq loss

    '''

    def __init__(self, data, history, item_num, max_len, weight, weightHis, neg_num,all_items):
        super().__init__()
        self.data = data
        self.history = history
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.weight = weight
        self.weightHis = weightHis
        self.var_name = ["seq", "pos", "neg", "positions","weight","history" ,"posHis", "negHis", "positionsHis","weightHis"]
        self.all_items = all_items

    def __len__(self):
        '''

        '''
        return len(self.data)
    
    def __getitem__(self, index):
        '''

        '''

        inter = self.data[index]

        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        weight = np.array(self.weight[index])

        nxt = inter[-1]

        idx = self.max_len - 1
        
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            # neg[idx] = random_select(self.all_items,non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:

            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:

            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)


        his = self.history[index]

        non_neg = copy.deepcopy(his)
        

        history = np.zeros([self.max_len], dtype=np.int32)
        posHis = np.zeros([self.max_len], dtype=np.int32)
        negHis = np.zeros([self.max_len], dtype=np.int32)
        weightHis = np.array(self.weightHis[index])

        nxt = his[-1]

        idx = self.max_len - 1
        
        for i in reversed(his[:-1]):
            history[idx] = i
            posHis[idx] = nxt
            negHis[idx] = random_neq(1, self.item_num+1, non_neg)
            # negHis[idx] = random_select(self.all_items,non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        if len(his) > self.max_len:

            mask_len = 0
            positionsHis = list(range(1, self.max_len+1))
        else:
            
            mask_len = self.max_len - (len(his) - 1)
            positionsHis = list(range(1, len(his)-1+1))
        
        positionsHis= positionsHis[-self.max_len:]
        positionsHis = [0] * mask_len + positionsHis
        positionsHis = np.array(positionsHis)

        return seq, pos, neg, positions, weight, history, posHis, negHis, positionsHis, weightHis
    
class LLMRecDataset(Dataset):
    '''

    '''
    def __init__(self, data, item_num, max_len,neg_num,args,all_items):
        
        super().__init__()
        self.data = data
        self.args = args
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions","posAug","negAug"]

        self.Aug = defaultdict(list) 
        f = open(f"data/{args.dataset}/LLMRec/interAugInter.txt", 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i1,i2 = line.rstrip().split(' ')
            u = int(u)
            i1 = int(i1)
            i2 = int(i2)
            self.Aug[u].append(i1)
            self.Aug[u].append(i2)
        self.all_items = all_items

    def __len__(self):
        '''

        '''
        return len(self.data)
    
    def __getitem__(self, index):
        '''

        '''

        inter = self.data[index]

        non_neg = copy.deepcopy(inter)
        

        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)

        posAug = np.zeros([self.max_len], dtype=np.int32)
        negAug = np.zeros([self.max_len], dtype=np.int32)

        nxt = inter[-1]

        idx = self.max_len - 1

        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            # neg[idx] = random_select(self.all_items,non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        nxt = self.Aug[idx][0] 

        idx = self.max_len - 1

        for i in reversed(inter[:-1]):
            posAug[idx] = nxt

            if i == inter[-2]:
                negAug[idx] = self.Aug[idx][1]
            else:
                negAug[idx] = random_neq(1, self.item_num+1, non_neg)

            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:

            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:

            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions,posAug,negAug