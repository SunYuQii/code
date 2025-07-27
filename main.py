'''

'''

import argparse
import torch
from utils.utils import set_seed
import os
from utils.logger import Logger
from generators.generator import Seq2SeqGenerator,AugSeqGenerator,ReliGenerator,LLMRecGenerator
from trainers.sequence_trainer import SeqTrainer
from models.confidenceTest import ConfidenceTest
from setproctitle import setproctitle

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", 
                    default='sasrec',
                    choices=[
                    "sasrec","aug_sasrec","aug_bert4rec","bert4rec","gru4rec","aug_gru4rec"
                    ],
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset",
                    default="fashion", 
                    # choices=["fashion","book","fashion5","yelp1205","book1205","book5","ASRepTrain","fashion10","book10","yelp10","fashion8"],  # preprocess by myself
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="reverseTrain",# ===================reverseTrain
                    choices=[
                    "inter","reverseInter","reverseInter_subset","inter_subset","reverseTrain","ASRepTrain"
                    ],
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--LLMRecPool",
                    default=False,#============================================================= 
                    action='store_true', 
                    help='whether generate LLMRec pool and prompt')
parser.add_argument("--LLMRec",
                    default=False,#============================================================= 
                    action='store_true', 
                    help='whether run LLMRec mode')
parser.add_argument("--ASRep",
                    default=False,#============================================================= 
                    action='store_true', 
                    help='whether generate ASRep pse')
parser.add_argument("--ASRepSave",
                    default=False,#============================================================= 
                    action='store_true', 
                    help='whether save the ASRep final aug result')
parser.add_argument("--demo",  # 
                    default=False,#============================================================= 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="sasrec_seq",
                    help="the path that pretrained model saved in")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="whehther run the test on the well-trained model")
parser.add_argument("--candidatePool",
                    default=False,# ======================================================false
                    action="store_true",
                    help="whehther run the well-trained SASRec model to get the condidate pool")
parser.add_argument("--poolSize",
                    type=int,
                    default=10,
                    help="the number of items in the condidate pool")
parser.add_argument("--pseudoNum",
                    type=int,
                    default=2,
                    help="the number of pseudo items")
# parser.add_argument("--LLM_model",
#                     default='gpt-3.5-turbo-0125',
#                     type=str,
#                     help="Large language model for pseudo-item generation")
parser.add_argument("--aug_file",
                    default="interAug",
                    type=str,
                    help="the name of enhanced interaction file")
parser.add_argument("--history_file",
                    default="inter",
                    type=str,
                    help="The file name to save the original sequence")
parser.add_argument("--ReliabilityTest",
                    default=False,# ===========================================================
                    action="store_true",
                    help="Reliability testing of enhancement results")
parser.add_argument("--userMask_file",
                    default="userMask",
                    type=str,
                    help="The file name to save the masked item")
parser.add_argument("--seqWeight_file",
                    default="weightone",
                    type=str,
                    help="the name of enhance sequence confidence file")
parser.add_argument("--use_aug",
                    default=False,# =========================================false
                    action="store_true",
                    help="whether to use augmented dataset training")
parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")
parser.add_argument("--ts_user",
                    type=int,
                    default=9,
                    help="the threshold to split the short and long seq")
parser.add_argument("--ts_item",
                    type=int,
                    default=4,
                    help="the threshold to split the long-tail and popular items")

parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="the number of GRU layers")

parser.add_argument("--train_batch_size",
                    default=128,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=1,# =================================================200
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=4,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=2, 
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=True,# ============================================================================default false
                    action="store_true",
                    help="whether create a new log file")


torch.autograd.set_detect_anomaly(True)
args = parser.parse_args()
set_seed(args.seed)

args.output_dir = os.path.join(args.output_dir, args.dataset)

args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)

args.output_dir = os.path.join(args.output_dir, args.model_name)

args.keepon_path = os.path.join(args.output_dir, args.keepon_path)

args.output_dir = os.path.join(args.output_dir, args.check_path)

if __name__ == "__main__":

    log_manager = Logger(args)
    # get the logger
    logger, writer = log_manager.get_logger()
    args.now_str = log_manager.get_now_str()
    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    # if args.model_name in ["llmesr_sasrec"]:
    #     generator = Seq2SeqGeneratorAllUser(args, logger, device)
    if args.model_name in ['sasrec','bert4rec','gru4rec'] and not args.ReliabilityTest and not args.LLMRec:
        '''

        '''
        generator = Seq2SeqGenerator(args, logger, device)
    elif args.model_name in ['aug_sasrec','aug_bert4rec','aug_gru4rec']:
        generator = AugSeqGenerator(args, logger, device)
    elif args.LLMRec:
        generator = LLMRecGenerator(args, logger, device)
    elif args.ReliabilityTest:

        conTest = ConfidenceTest(args)
        conTest.getMaskItem()
        generator = ReliGenerator(args,logger,device)
    else:
        raise ValueError
    
    trainer = SeqTrainer(args, logger, writer, device, generator)

    if args.do_test:
        trainer.test_group()
    elif args.candidatePool:
        trainer.getCandidatePool('pse_candidate.bin')
    elif args.ReliabilityTest:
        # Reliability testing

        trainer.getCandidatePool('rel_candidate.bin')
    elif args.LLMRecPool:
        trainer.getCandidatePool('rel_candidate.bin')
    elif args.use_aug:

        trainer.augTrain()
        trainer.test_group()
    elif args.LLMRec:
        trainer.LLMRecTrain()
    elif args.ASRep:
        trainer.getASRepPse()
    else:
        trainer.train()

    # # delete the logger threads
    log_manager.end_log()
