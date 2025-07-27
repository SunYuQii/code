'''

'''

from collections import defaultdict
import random
import json
import copy
import os

class ConfidenceTest(object):
    def __init__(self, args):
        self.args = args
        # self.prompt_template = "The user has visited following fashions:\n<HISTORY> Please observe the user's historical interactions and fill in the appropriate item(only one) at <MASK>. Please do not reply with any information other than the item name."

    def getMaskItem(self):
        '''

        '''


        UserAug = defaultdict(list)    # default value is a blank list
        f = open('./data/%s/handled/%s.txt' % (self.args.dataset,self.args.aug_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            UserAug[u].append(i)

        UserHis = defaultdict(list)    # default value is a blank list 
        # beforeAug
        f = open('./data/%s/handled/%s.txt' % (self.args.dataset,self.args.history_file), 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            UserHis[u].append(i)

        id_map = json.load(open("data/{}/handled/id_map.json".format(self.args.dataset)))

        item_dict = json.load(open("data/{}/handled/item2attributes.json".format(self.args.dataset), "r"))

        userLabel = {}
        userTitle = {}
        userMaskIdx = {}
        if "yelp" in self.args.dataset.lower():
            keyword = "name"
        else:
            keyword = "title"

        random.seed(self.args.seed)
        for usr in UserHis.keys():

            while(True):
                
                label = UserHis[usr][-2]
                
                item = id_map['id2item'][str(label)]
                try:

                    userTitle[usr] = item_dict[item][keyword].replace("\n","")
                    break
                except:
                    continue
            index = -1
            for idx,elem in enumerate(reversed(UserHis[usr])):
                if elem == label:
                    userMaskIdx[usr] = idx+1
                    break
            userLabel[usr] = label
            # print(f"seq is {UserHis[usr]},choice is {label},pos is{reverse_position},test{UserHis[usr][0-reverse_position]}")
        User = {'userLable':userLabel,'userTitle':userTitle}

        if not os.path.exists("data//%s/reliability"% (self.args.dataset)):

            os.makedirs("data//%s/reliability"% (self.args.dataset))
        json.dump(User, open("data//%s/reliability/%s.json"% (self.args.dataset,self.args.userMask_file), "w"))


        Aug2Title = defaultdict(list)
        getMaskP = {}
        for usr in UserAug.keys():
            for index,item in enumerate(UserAug[usr]):

                if len(UserAug[usr])-index == userMaskIdx[usr]:
                    Aug2Title[usr].append("<MASK>")
                    UserAug[usr][index] = 0
                else:
                    item = id_map['id2item'][str(item)]
                    try:
                        Aug2Title[usr].append(item_dict[item][keyword].replace("\n",""))
                    except:
                        continue

        json.dump(UserAug,open("data/{}/reliability/maskedAug.json".format(self.args.dataset), "w"))
            