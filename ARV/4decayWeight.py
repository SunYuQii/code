'''
input file: weight2.txt
output file: weightDecay0-5.txt
'''

from collections import defaultdict
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    default="fashion", 
                    help="Choose the dataset")
parser.add_argument("--pseNum",
                    default=12,
                    type=int,
                    help="the number of the pse num")
parser.add_argument("--T",
                    default=4,
                    type=int,
                    help="the decayThreshold")
parser.add_argument("--decayRate",
                    default=0.5,
                    type=float,
                    help="the decay Rate")
args = parser.parse_args()
print(args)

dataset = args.dataset

# fashion-4
# book1205-7
decayThreshold = args.T
pseNum = args.pseNum
decayThreshold = decayThreshold+pseNum
decayRate = args.decayRate
user2Weight = {}

f = open('./data/%s/handled/weight%s.txt' % (dataset,str(pseNum)), 'r')
for line in f:  # use a dict to save all seqeuces of each user
    u, weight = line.rstrip().split(' ')
    u = int(u)
    weight = float(weight)
    user2Weight[u] = weight


f = open('./data/%s/handled/%s.txt' % (dataset,"interAug"), 'r')
User = defaultdict(list)    # default value is a blank list
for line in f:  # use a dict to save all seqeuces of each user
    u, i = line.rstrip().split(' ')
    u = int(u)
    i = int(i)
    User[u].append(i)

with open(f"data/{dataset}/handled/weightDecay{decayRate}.txt", 'w') as f:
    for user in tqdm(user2Weight.keys()):
        u = int(user)
        if len(User[u])<= decayThreshold:
            weight = round(user2Weight[user],6)
        else:
            weight = round(decayRate*user2Weight[user],6)
        f.write('%s %s\n' % (u, weight))
