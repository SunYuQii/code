'''
'''

from collections import defaultdict
from tqdm import tqdm
from trainers.trainer import Trainer
import torch
import os
from utils.utils import metric_report,metric_len_5group,metric_len_3group,metric_len_4group,metric_pop_5group,metric_report,metric_len_report,metric_pop_report,record_csv
import time
import json
import copy
import re

def extract_prefix(dataset):
    match = re.match(r"^[a-zA-Z]+", dataset)  
    if match:
        return match.group(0)  
    return None

class SeqTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
        self.candT2id = {}

    def _train_one_epoch(self, epoch):


        tr_loss = 0

        nb_tr_examples, nb_tr_steps = 0, 0

        train_time = []


        self.model.train()
        
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            inputs = self._prepare_train_inputs(batch)

            loss = self.model(**inputs)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss

            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))


            self.optimizer.step()

            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)


        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)

    def eval(self, epoch=0, test=False):
        '''

        '''
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader

        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])

            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict5 = metric_report(pred_rank.detach().cpu().numpy(),5)
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_dict20 = metric_report(pred_rank.detach().cpu().numpy(),20)
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(),args=self.args)
        res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        for k, v in res_dict5.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        for k, v in res_dict20.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))

        if test:
            self.logger.info("User Group Performance:")
            for k, v in res_len_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
            self.logger.info("Item Group Performance:")
            for k, v in res_pop_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict, **res_pop_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict

    def test_group(self):
        '''

        Do test directly. Set the output dir as the path that save the checkpoint
        '''
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running Group test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)

        test_loader = self.test_loader
        self.model.eval()

        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)


        for batch in tqdm(test_loader, desc=desc):


            batch = tuple(t.to(self.device) for t in batch)

            inputs = self._prepare_eval_inputs(batch)


            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])

            target_items = torch.cat([target_items, inputs["pos"]])
            

            with torch.no_grad():

                # shape [batch,1]+[batch,neg_num]=[batch,neg_num+1]
                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                
                # [batch,neg_num+1]
                pred_logits = -self.model.predict(**inputs)


                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)

        if "fashion" in self.args.dataset:
            UserGroup = [2.5,3.5]
        elif "book" in self.args.dataset:
            UserGroup = [2.5,4.5]
        elif "yelp" in self.args.dataset:
            UserGroup = [5.5,7.5]
        if self.args.use_aug:
            for idx,value in enumerate(UserGroup):
                UserGroup[idx] = value + self.args.pseudoNum
        print(UserGroup)
        # hr_len, ndcg_len, count_len = metric_len_5group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), UserGroup)
        hr_len, ndcg_len, count_len = metric_len_3group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), UserGroup)
        # hr_pop, ndcg_pop, count_pop = metric_pop_5group(pred_rank.detach().cpu().numpy(), self.item_pop,  target_items.detach().cpu().numpy(), [10, 30, 60, 100])
        # if "book" in self.args.dataset:
        #     UserGroup = [3.5,6.5,9.5]
        #     hr_len, ndcg_len, count_len = metric_len_4group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), UserGroup)
        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("User Group Performance:")
        for i, (hr, ndcg, num) in enumerate(zip(hr_len, ndcg_len,count_len)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f, num %.4f' % (i, hr, ndcg, num))
        # self.logger.info("Item Group Performance:")
        # for i, (hr, ndcg) in enumerate(zip(hr_pop, ndcg_pop)):
        #     self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        
        
        return res_dict
    
    def get_candidate_prompt(self,history,candidate):
        '''

        '''
        if "yelp" in self.args.dataset.lower():
            keyword = "name"
        else:
            keyword = "title"
        candidate_str = copy.deepcopy(self.prompt_template)
        hist_str = ""
        cand_str= ""
        for item in history:
            try:    
                # some item does not have title,for example 0992916305,1620213982,'014789302X'

                item_str = self.item_dict[item][keyword].replace("\n","")
                hist_str = hist_str + item_str + "\n"
            except:
                continue

        hist_str = hist_str[:-1]
        # limit the prompt length
        if len(hist_str) > 8000:
            hist_str = hist_str[-8000:]
        candidate_str = candidate_str.replace("<HISTORY>", hist_str)
        candT2id = {}
        for item in candidate:
            try:    
                # some item does not have title

                item_str = self.item_dict[item][keyword].replace("\n","")
                candT2id[item_str] = item
                cand_str = cand_str + item_str + "\n"
            except:
                continue
        # limit the prompt length
        if len(cand_str) > 8000:
            cand_str = cand_str[-8000:]
        candidate_str = candidate_str.replace("<CANDIDATE>", cand_str)

        return candidate_str,cand_str.split('\n'),candT2id
    
    def getCandidatePool(self,modelPath):
        '''

        '''
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info(f"********** Running candidate pool generate using {modelPath} **********")
        desc = 'candidate item'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, modelPath))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)

        test_loader = self.candidate_loader
        self.model.eval()

        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        # user-candidateItem
        u2cand = {}
        # user-history
        u2history = {}


        for batch in tqdm(test_loader, desc=desc):


            batch = tuple(t.to(self.device) for t in batch)

            inputs = self._prepare_candidate_inputs(batch)

            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])

            target_items = torch.cat([target_items, inputs["pos"]])
            

            with torch.no_grad():

                # shape [batch,1]+[batch,neg_num]=[batch,neg_num+1]
                # inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1) # 这里不需要两个pos item
                inputs["item_indices"] =  inputs["neg"]
                pred_logits = -self.model.predict(**inputs) 

                per_pred_rank_all = torch.argsort(torch.argsort(pred_logits))
                per_pred_rank = per_pred_rank_all[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])


                mask = per_pred_rank_all<self.args.poolSize
                candidateItem = inputs["item_indices"][mask]
                # [batch,args.poolSize]
                candidateItem = candidateItem.reshape([mask.shape[0],-1])
                for idx,user in enumerate(inputs["user_id"]):
                    # u2cand[user.item()] = [elem.item() for elem in candidateItem[idx]]
                    u2cand[user.item()] = candidateItem[idx]

                    mask = inputs["seq"][idx]!=0
                    # u2history[user.item()] = [elem.item() for elem in inputs["seq"][idx][mask]]
                    u2history[user.item()] = inputs["seq"][idx][mask]



        # self.args.candidatePath = os.path.join(self.args.output_dir, self.args.candidatePath)
        # candidate_path = os.path.join(self.args.candidatePath, 'candidate.json')
        # candidateMap = {"u2cand":u2cand,"u2history":u2history}

        id_map = json.load(open("data/{}/handled/id_map.json".format(self.args.dataset)))
        # print(id_map["id2item"])
        user2history = {}
        user2candidate = {}
        for usr in u2cand.keys():
            user2history[usr] = [id_map["id2item"][str(elem.item())] for elem in u2history[usr]]
            user2candidate[usr] = [id_map["id2item"][str(elem.item())] for elem in u2cand[usr]]

        self.item_dict = json.load(open("data/{}/handled/item2attributes.json".format(self.args.dataset), "r"))


        candidate_prompt = {}
        cand_list = {}
        candidate_data = {}
        pseudoitem_data = {}
        aug_data = {}
        name = ""
        if "yelp" in self.args.dataset.lower():
            name = "item"
        else:
            name = extract_prefix(self.args.dataset)
        if self.args.candidatePool:
            self.prompt_template = "The user has visited following {}s: \n<HISTORY> \nPlease observe the user's historical interactions and select {} items that the user may interact with from the candidate set: \n<CANDIDATE>.Please use \n to separate items.Please reply with only the name of the {} items, no other information.".format(name,self.args.pseudoNum,self.args.pseudoNum)
        elif self.args.ReliabilityTest:
            self.prompt_template = "The user has visited following {}s: \n<HISTORY> \nPlease observe the user's historical interactions and select {} item that the user may interact with from the candidate set: \n<CANDIDATE>.Please reply with only one item name.".format(name,"one")
        elif self.args.LLMRecPool:
            self.prompt_template = "The user has visited following {}s: \n<HISTORY> \n\nPlease observe the user's historical interactions and select one user\'s favorite item and one least favorite item from the candidate set: \n<CANDIDATE>.Please use \n to separate items.Nothing else.Plese just give the title of items.".format(name)
        for usr in user2history.keys():
            candidate_prompt[usr],cand_list[usr],cand2id_user = self.get_candidate_prompt(user2history[usr],user2candidate[usr])

            for cand in cand2id_user.keys():
                cand2id_user[cand] = id_map["item2id"][cand2id_user[cand]]
            self.candT2id[usr] = cand2id_user

            if cand_list[usr][-1] == '':
                cand_list[usr] = cand_list[usr][:-1]
        if self.args.candidatePool:

            if not os.path.exists("data/{}/pseudo".format(self.args.dataset)):

                os.makedirs("data/{}/pseudo".format(self.args.dataset))
            json.dump(candidate_prompt, open("data/{}/pseudo/candidate_prompt.json".format(self.args.dataset), "w"))
            json.dump(cand_list, open("data/{}/pseudo/cand_list.json".format(self.args.dataset), "w"))
            json.dump(self.candT2id, open("data/{}/pseudo/candT2id.json".format(self.args.dataset), "w"))
        elif self.args.ReliabilityTest:

            if not os.path.exists("data/{}/reliability".format(self.args.dataset)):

                os.makedirs("data/{}/reliability".format(self.args.dataset))
            json.dump(candidate_prompt, open("data/{}/reliability/candidate_prompt.json".format(self.args.dataset), "w"))
            json.dump(cand_list, open("data/{}/reliability/cand_list.json".format(self.args.dataset), "w"))
            json.dump(self.candT2id, open("data/{}/reliability/candT2id.json".format(self.args.dataset), "w"))
        elif self.args.LLMRecPool:

            if not os.path.exists("data/{}/LLMRec".format(self.args.dataset)):

                os.makedirs("data/{}/LLMRec".format(self.args.dataset))
            json.dump(candidate_prompt, open("data/{}/LLMRec/candidate_prompt.json".format(self.args.dataset), "w"))
            json.dump(cand_list, open("data/{}/LLMRec/cand_list.json".format(self.args.dataset), "w"))
            json.dump(self.candT2id, open("data/{}/LLMRec/candT2id.json".format(self.args.dataset), "w"))
        
        self.logger.info("********** candidate item generate finish **********")


    def _trainAug_one_epoch(self, epoch):

        tr_loss = 0

        nb_tr_examples, nb_tr_steps = 0, 0

        train_time = []


        self.model.train()

        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()

            inputs,Hisinputs = self._prepare_train_Auginputs(batch)

            lossAug = self.model(**inputs)
            lossHis = self.model(**Hisinputs)
            lossFinal = lossAug+lossHis
            lossFinal.backward()

            tr_loss += lossFinal.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss

            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))


            self.optimizer.step()

            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)


        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)

    def _trainLLMRec_one_epoch(self,epoch):

        tr_loss = 0

        nb_tr_examples, nb_tr_steps = 0, 0

        train_time = []


        self.model.train()

        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()

            inputs,LLMRecAug = self._prepare_train_LLMRecinputs(batch)

            lossAug = self.model(**inputs)
            lossLLMRec = self.model(**LLMRecAug)
            lossFinal = lossAug + lossLLMRec
            lossFinal.backward()

            tr_loss += lossFinal.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss

            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))


            self.optimizer.step()

            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)


        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)

    def getASRepPse(self):
        '''

        '''
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running ASRep pse generate **********")
        desc = 'candidate item'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, "pse_candidate.bin"))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)

        test_loader = self.candidate_loader
        self.model.eval()

        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        u2pse = {}
        u2history = {}
        u2aug = {}
        User = defaultdict(list)

        if self.args.ASRepSave:
            f = open('./data/%s/handled/%s.txt' % (self.args.dataset, self.args.history_file), 'r')
            for line in f:  # use a dict to save all seqeuces of each user
                u, i = line.rstrip().split(' ')
                u = int(u)
                i = int(i)
                User[u].append(i)


        for batch in tqdm(test_loader, desc=desc):


            batch = tuple(t.to(self.device) for t in batch)

            inputs = self._prepare_candidate_inputs(batch)


            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])

            target_items = torch.cat([target_items, inputs["pos"]])


            with torch.no_grad():
 
                # shape [batch,1]+[batch,neg_num]=[batch,neg_num+1]
                # inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1) # 这里不需要两个pos item
                inputs["item_indices"] =  inputs["neg"]
                pred_logits = -self.model.predict(**inputs) 

                per_pred_rank_all = torch.argsort(torch.argsort(pred_logits))
                per_pred_rank = per_pred_rank_all[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])


                mask = per_pred_rank_all<1
                candidateItem = inputs["item_indices"][mask]
                # [batch,1]
                candidateItem = candidateItem.reshape([mask.shape[0],-1])
                for idx,user in enumerate(inputs["user_id"]):
                    u2pse[user.item()] = candidateItem[idx]

                    mask = inputs["seq"][idx]!=0
                    u2history[user.item()] = inputs["seq"][idx][mask]
                    if self.args.ASRepSave:
                        u2aug[user.item()] = list(u2pse[user.item()]) + list(reversed(list(u2history[user.item()])))
                        u2aug[user.item()].append(User[user.item()][-1])
                    else:
                        u2aug[user.item()] = list(u2history[user.item()])+list(u2pse[user.item()])
        

        if self.args.ASRepSave:
            # if not os.path.exists("data/{}/ASRep".format(self.args.dataset)):

            #     os.makedirs("data/{}/ASRep".format(self.args.dataset))
            with open("data/{}/handled/ASRepAug.txt".format(self.args.dataset), "w") as f:
                for user,item_list in tqdm(u2aug.items()):
                    for item in item_list:
                        u = int(user)
                        i = int (item)
                        f.write('%s %s\n' % (u, i))
        else:
            # if not os.path.exists("data/{}/ASRep".format(self.args.dataset)):

            #     os.makedirs("data/{}/ASRep".format(self.args.dataset))
            with open("data/{}/handled/ASRepTrain.txt".format(self.args.dataset), "w") as f:
                for user,item_list in tqdm(u2aug.items()):
                    for item in item_list:
                        u = int(user)
                        i = int (item)
                        f.write('%s %s\n' % (u, i))
