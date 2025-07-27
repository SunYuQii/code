from collections import defaultdict
import time
from tqdm import tqdm
from utils.utils import unzip_data,concat_data
from generators.data import SeqDataset,Seq2SeqDataset,Seq2SeqDatasetAllUser,AugSeqDataset,LLMRecDataset
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
import numpy as np
import json

class Generator(object):

    def __init__(self, args, logger, device):

        self.args = args
        # self.aug_file = args.aug_file
        self.inter_file = args.inter_file
        self.dataset = args.dataset
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device
        # self.aug_seq = args.aug_seq

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    def _load_dataset(self):
        '''
        Load train, validation, test dataset
        '''

        usernum = 0
        itemnum = 0
        User = defaultdict(list)    # default value is a blank list
        user_train = {}
        user_valid = {}
        user_test = {}
        itemsApp = {}
        # assume user/item index starting from 1
        f = open('./data/%s/handled/%s.txt' % (self.dataset, self.inter_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
            if i in itemsApp.keys():
                continue
            else:
                itemsApp[i] = 1
        
        self.user_num = usernum
        self.item_num = itemnum
        # id_map = json.load(open(f"data/{self.dataset}/handled/id_map.json", "r"))
        # self.item_num = len(id_map['id2item'])
        self.all_items = list(itemsApp.keys())
        # print(len(self.all_items),self.item_num)


        for user in tqdm(User):
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        
        self.train = user_train
        self.valid = user_valid
        self.test = user_test

    def make_evalloader(self, test=False):
        '''

        '''
        if test:
            eval_dataset = concat_data([self.train, self.valid, self.test])

        else:
            eval_dataset = concat_data([self.train, self.valid])

        self.eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg,self.all_items)
        eval_dataloader = DataLoader(self.eval_dataset,
                                    sampler=SequentialSampler(self.eval_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return eval_dataloader

    def get_user_item_num(self):
        '''

        '''
        return self.user_num, self.item_num
    
    def get_item_pop(self):
        """
        get item popularity according to item index. return a np-array
        """
        all_data = concat_data([self.train, self.valid, self.test])
        # item index starts from 0
        pop = np.zeros(self.item_num+1) 
        
        for items in all_data:
            pop[items] += 1

        return pop
    
    def get_user_len(self):
        """
        get (train+valid)sequence length according to user index. return a np-array
        """
        all_data = concat_data([self.train, self.valid])
        lens = []

        for user in all_data:
            lens.append(len(user))

        return np.array(lens)

'''

'''
class Seq2SeqGenerator(Generator):
    '''

    '''
    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)

    def make_trainloader(self):
        '''

        '''
        train_dataset = unzip_data(self.train)

        # cnt = 0
        # for elem in train_dataset:
        #     if len(elem) > 16:
        #         cnt = cnt+1
        # print(cnt/len(train_dataset))
        self.train_dataset = Seq2SeqDataset(train_dataset, self.item_num, self.args.max_len, self.args.train_neg,self.all_items)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader
    
    def make_candidateloader(self):
        '''

        '''

        # alls_dataset_label = concat_data([self.train, self.valid, self.test])
        # with open(f"data/{self.args.dataset}/handled/{self.args.history_file}.txt", 'w') as f:
        #     for user,item_list in enumerate(alls_dataset_label):

        #         for item in reversed(item_list):
        #             u = int(user)
        #             i = int(item)
        #             f.write('%s %s\n' % (u, i))
        # for user in self.train.keys():
        #     self.train[user] = self.train[user][1:]
        alls_dataset = concat_data([self.train, self.valid, self.test])
        self.alls_dataset = Seq2SeqDatasetAllUser(alls_dataset, self.item_num, self.args.max_len, self.all_items, self.args.test_neg)
        alls_dataloader = DataLoader(self.alls_dataset,
                                    sampler=SequentialSampler(self.alls_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return alls_dataloader

class AugSeqGenerator(object):
    '''

    '''
    def __init__(self, args, logger, device):
        self.args = args
        self.inter_file = args.inter_file
        self.dataset = args.dataset
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device


        self.logger.info("Loading augmented dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Augmented dataset is loaded: consume %.3f s" % (end - start))

    def _load_dataset(self):
        '''
        Load train, validation, test dataset
        '''


        usernum = 0
        itemnum = 0
        User = defaultdict(list)    # default value is a blank list
        user_train = {}
        user_valid = {}
        user_test = {}

        user_weight = {}
        user_weightHis = {}

        itemsApp = {}
        # assume user/item index starting from 1
        f = open('./data/%s/handled/%s.txt' % (self.dataset,self.args.aug_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
            if i in itemsApp.keys():
                continue
            else:
                itemsApp[i] = 1
        
        self.user_num = usernum
        self.item_num = itemnum
        # self.item_num = len(itemsApp)
        self.all_items = list(itemsApp.keys())

        # max_value = max(max(values) for values in User.values() if values) 
        for user in tqdm(User):
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        
        self.Augtrain = user_train
        self.Augvalid = user_valid
        self.Augtest = user_test

        User = defaultdict(list)    # default value is a blank list
        user_train = {}
        f = open('./data/%s/handled/%s.txt' % (self.dataset,self.args.history_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            User[u].append(i)


        for user in tqdm(User):
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
            else:
                user_train[user] = User[user][:-2]
        
        self.train = user_train

        f = open('./data/%s/handled/%s.txt' % (self.dataset,self.args.seqWeight_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, weight = line.rstrip().split(' ')
            u = int(u)
            weight = float(weight)
            user_weight[u] = weight
            user_weightHis[u] = 1-weight

        self.user_weight = user_weight
        self.user_weightHis = user_weightHis

    def get_user_item_num(self):
        '''

        '''
        return self.user_num, self.item_num
    
    def get_item_pop(self):
        """
        get item popularity according to item index. return a np-array

        """
        all_data = concat_data([self.Augtrain, self.Augvalid, self.Augtest])
        # item index starts from 0
        pop = np.zeros(self.item_num+1)
        
        for items in all_data:
            pop[items] += 1

        return pop
    
    def get_user_len(self):
        """
        get (train+valid)sequence length according to user index. return a np-array

        """
        all_data = concat_data([self.Augtrain, self.Augvalid])
        lens = []

        for user in all_data:
            lens.append(len(user))

        return np.array(lens)

    def make_trainloader(self):
        '''

        '''

        assert len(self.train) == len(self.user_weightHis)
        assert len(self.Augtrain) == len(self.user_weight)
        train_dataset = unzip_data(self.train)
        Augtrain_dataset = unzip_data(self.Augtrain)

        user_weight = unzip_data(self.user_weight)
        user_weightHis = unzip_data(self.user_weightHis)
        self.Augtrain_dataset = AugSeqDataset(Augtrain_dataset, train_dataset, self.item_num, self.args.max_len, user_weight, user_weightHis, self.args.train_neg,self.all_items)

        train_dataloader = DataLoader(self.Augtrain_dataset,
                                      sampler=RandomSampler(self.Augtrain_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader
    
    def make_evalloader(self, test=False):
        '''

        '''
        if test:
            eval_dataset = concat_data([self.Augtrain, self.Augvalid, self.Augtest])

        else:
            eval_dataset = concat_data([self.Augtrain, self.Augvalid])

        self.eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg,self.all_items)
        eval_dataloader = DataLoader(self.eval_dataset,
                                    sampler=SequentialSampler(self.eval_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return eval_dataloader
    
class ReliGenerator(Generator):
    '''

    '''
    def __init__(self, args, logger, device):
        self.args = args
        self.logger = logger
        self.device = device


        self.logger.info("Loading masked augment dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Augmented dataset is loaded: consume %.3f s" % (end - start))

    def _load_dataset(self):
        '''
        Load train, validation, test dataset
        '''

        usernum = 0
        itemnum = 0
        User = defaultdict(list)    # default value is a blank list

        itemsApp = {}

        f = open('./data/%s/handled/%s.txt' % (self.args.dataset, self.args.inter_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)

        # assume user/item index starting from 1

        f = open('./data/%s/reliability/%s.json' % (self.args.dataset,"maskedAug"), 'r')
        maskedAug = json.load(f)
        for user in maskedAug.keys():
            for elem in maskedAug[user]:
                if int(elem) == 0:
                    break
                User[user].append(int(elem))
                if elem not in itemsApp.keys():
                    itemsApp[elem] = 1
        self.all_items = list(itemsApp.keys()) 
        self.User = User
        self.user_num = usernum

        # print(self.user_num,User.keys())
        self.item_num = itemnum

        # id_map = json.load(open(f"data/{self.args.dataset}/handled/id_map.json", "r"))
        # self.item_num = len(id_map['id2item'])

    def make_candidateloader(self):
        '''

        '''

        alls_dataset = []
        for user in self.User.keys():
            alls_dataset.append(self.User[user])
        self.alls_dataset = Seq2SeqDatasetAllUser(alls_dataset, self.item_num, self.args.max_len, self.all_items, self.args.test_neg)
        alls_dataloader = DataLoader(self.alls_dataset,
                                    sampler=SequentialSampler(self.alls_dataset),
                                    batch_size=100,
                                    num_workers=self.args.num_workers)
        
        return alls_dataloader
    
class LLMRecGenerator(Generator):
    '''
    '''
    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)

    def make_trainloader(self):
        '''
        '''
        train_dataset = unzip_data(self.train)
        self.train_dataset = LLMRecDataset(train_dataset, self.item_num, self.args.max_len, self.args.train_neg,self.args,self.all_items)
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader