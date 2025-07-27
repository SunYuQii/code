'''
'''

from models.SASRec import SASRec,SASRecAug
from models.Bert4Rec import Bert4Rec,Bert4RecAug
from models.GRU4Rec import GRU4Rec,GRU4RecAug
from utils.utils import get_n_params
import torch
from utils.earlystop import EarlyStopping
import os
from tqdm import trange
from models.confidenceTest import ConfidenceTest

class Trainer(object):
    def __init__(self, args, logger, writer, device, generator):

        self.args = args
        self.logger = logger
        self.writer = writer
        self.device = device
        self.user_num, self.item_num = generator.get_user_item_num()
        self.start_epoch = 0    # define the start epoch for keepon training

        self.logger.info('Loading Model: ' + args.model_name)
        self._create_model()
        logger.info('# of model parameters: ' + str(get_n_params(self.model)))
        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()

        if args.keepon:
            self._load_pretrained_model()

        self.loss_func = torch.nn.BCEWithLogitsLoss()


        if self.args.candidatePool or self.args.ReliabilityTest or self.args.LLMRecPool or self.args.ASRep:
            self.candidate_loader = generator.make_candidateloader()
        else:
            self.train_loader = generator.make_trainloader()
            self.valid_loader = generator.make_evalloader()
            self.test_loader = generator.make_evalloader(test=True)

            # get item pop and user len
            self.item_pop = generator.get_item_pop()
            self.user_len = generator.get_user_len()

        self.generator = generator
        self.watch_metric = args.watch_metric

    def _create_model(self):
        '''
        create your model
        '''
        if self.args.model_name in ["aug_sasrec"]:
            self.model = SASRecAug(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name in ["sasrec"]:
            self.model = SASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name in ["bert4rec"]:
            self.model = Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name in ["aug_bert4rec"]:
            self.model = Bert4RecAug(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name in ["gru4rec"]:
            self.model = GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name in ["aug_gru4rec"]:
            self.model = GRU4RecAug(self.user_num, self.item_num, self.device, self.args)
        else:
            raise ValueError
        
        self.model.to(self.device)

    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.args.lr,
                                          weight_decay=self.args.l2,
                                          )
        
    def _set_scheduler(self):

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.args.lr_dc_step,
                                                         gamma=self.args.lr_dc)
        
    def _set_stopper(self):

        self.stopper = EarlyStopping(patience=self.args.patience, 
                                     verbose=False,
                                     path=self.args.output_dir,
                                     trace_func=self.logger)
        
    def _load_pretrained_model(self):

        self.logger.info("Loading the trained model for keep on training ... ")
        checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.bin')

        model_dict = self.model.state_dict()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']

        # filter out required parameters
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # Print: how many parameters are loaded from the checkpoint
        self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        self.model.load_state_dict(model_dict)  # load model parameters
        self.optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
        self.scheduler.load_state_dict(checkpoint['scheduler']) # load scheduler
        self.start_epoch = checkpoint['epoch']  # load epoch

    def _prepare_train_inputs(self, data):
        """Prepare the inputs as a dict for training"""
        assert len(self.generator.train_dataset.var_name) == len(data)
        inputs = {}
        for i, var_name in enumerate(self.generator.train_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs
    
    def _prepare_train_Auginputs(self, data):
        """Prepare the inputs as a dict for Aug training"""
        assert len(self.generator.Augtrain_dataset.var_name) == len(data)
        inputs = {}
        Hisinputs = {}
        limit = int(len(self.generator.Augtrain_dataset.var_name)/2)
        for i, var_name in enumerate(self.generator.Augtrain_dataset.var_name):
            if i<limit:
                inputs[var_name] = data[i]
            else:
                Hisinputs[self.generator.Augtrain_dataset.var_name[i%limit]] = data[i]

        return inputs,Hisinputs

    def _prepare_eval_inputs(self, data):
        """
        Prepare the inputs as a dict for evaluation
        return:
        'seq' =
        tensor([[  0,   0,   0,  ...,   1,   2,   2],
                [  0,   0,   0,  ...,   0,   4,   5],
                [  0,   0,   0,  ...,   0,   8,   8],
                ...,
                [  0,   0,   0,  ...,   0, 160,  23],
                [  0,   0,   0,  ...,   0, 161,  23],
                [  0,   0,   0,  ...,   0,  23,  24]], device='cuda:0',
            dtype=torch.int32)
        'pos' =
        tensor([[  0,   0,   0,  ...,   2,   2,   3],
                [  0,   0,   0,  ...,   0,   5,   6],
                [  0,   0,   0,  ...,   0,   8,   5],
                ...,
                [  0,   0,   0,  ...,   0,  23,  24],
                [  0,   0,   0,  ...,   0,  23,  24],
                [  0,   0,   0,  ...,   0,  24, 162]], device='cuda:0',
            dtype=torch.int32)
        'neg' =
        tensor([[   0,    0,    0,  ...,   26,  463, 3394],
                [   0,    0,    0,  ...,    0, 3212,  627],
                [   0,    0,    0,  ...,    0, 1033, 1279],
                ...,
                [   0,    0,    0,  ...,    0, 2995, 1633],
                [   0,    0,    0,  ...,    0,  990, 2815],
                [   0,    0,    0,  ...,    0, 2295,  141]], device='cuda:0',
            dtype=torch.int32)
        'positions' =
        tensor([[0, 0, 0,  ..., 1, 2, 3],
        """
        inputs = {}
        assert len(self.generator.eval_dataset.var_name) == len(data)
        for i, var_name in enumerate(self.generator.eval_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs
    
    def _prepare_candidate_inputs(self, data):
        '''

        '''
        inputs = {}
        assert len(self.generator.alls_dataset.var_name) == len(data)
        for i, var_name in enumerate(self.generator.alls_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs
    
    def _prepare_train_LLMRecinputs(self,data):
        '''
        1 Prepare the inputs as a dict for training
        '''
        assert len(self.generator.train_dataset.var_name) == len(data)
        inputs = {}
        Auginputs = {}
        for i, var_name in enumerate(self.generator.train_dataset.var_name):
            if var_name in ["seq","pos","neg","positions"]:
                inputs[var_name] = data[i]
            if var_name in ["seq","posAug","negAug","positions"]:
                if var_name in ["posAug"]:
                    Auginputs["pos"] = data[i]
                elif var_name in ["negAug"]:
                    Auginputs["neg"] = data[i]
                else:
                    Auginputs[var_name] = data[i]
        return inputs,Auginputs
    
    def train(self):
        '''

        '''
        # Only save the model it-self
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        res_list = []
        train_time = []

        # epoch
        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.num_train_epochs), desc="Epoch"):
            t = self._train_one_epoch(epoch)
            train_time.append(t)
             # evluate on validation per epochs
            if (epoch % 1) == 0:
                metric_dict = self.eval(epoch=epoch)
                res_list.append(metric_dict)
                self.stopper(metric_dict[self.watch_metric], epoch, model_to_save, self.optimizer, self.scheduler)
                if self.stopper.early_stop:
                    break

        best_epoch = self.stopper.best_epoch
        best_res = res_list[best_epoch - self.start_epoch]
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)
        self.logger.info('The best results are NDCG@10: %.5f, HR@10: %.5f' %
                    (best_res['NDCG@10'], best_res['HR@10']))
        
        res = self.eval(test=True)

        return res, best_epoch

    def augTrain(self):
        # Only save the model it-self
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        res_list = []
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.num_train_epochs), desc="Epoch"):
            t = self._trainAug_one_epoch(epoch)
            train_time.append(t)
            # evluate on validation per epochs
            if (epoch % 1) == 0:
                metric_dict = self.eval(epoch=epoch)
                res_list.append(metric_dict)
                self.stopper(metric_dict[self.watch_metric], epoch, model_to_save, self.optimizer, self.scheduler)
                if self.stopper.early_stop:
                    break

        best_epoch = self.stopper.best_epoch
        best_res = res_list[best_epoch - self.start_epoch]
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)
        self.logger.info('The best results are NDCG@10: %.5f, HR@10: %.5f' %
                    (best_res['NDCG@10'], best_res['HR@10']))
        
        res = self.eval(test=True)

        return res, best_epoch
    
    def LLMRecTrain(self):
        '''

        '''
        # Only save the model it-self
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        res_list = []
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.num_train_epochs), desc="Epoch"):
            t = self._trainLLMRec_one_epoch(epoch)
            train_time.append(t)
            # evluate on validation per epochs
            if (epoch % 1) == 0:
                metric_dict = self.eval(epoch=epoch)
                res_list.append(metric_dict)
                self.stopper(metric_dict[self.watch_metric], epoch, model_to_save, self.optimizer, self.scheduler)
                if self.stopper.early_stop:
                    break

        best_epoch = self.stopper.best_epoch
        best_res = res_list[best_epoch - self.start_epoch]
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)
        self.logger.info('The best results are NDCG@10: %.5f, HR@10: %.5f' %
                    (best_res['NDCG@10'], best_res['HR@10']))
        
        res = self.eval(test=True)

        return res, best_epoch
