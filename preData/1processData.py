'''
'''
import numpy as np
import gzip
from tqdm import tqdm
import json
from collections import defaultdict
import os

def parse(path):
    '''
    '''
    g = gzip.open(path, 'rb')
    inter_list = []
    for l in tqdm(g):
        inter_list.append(json.loads(l.decode()))
    return inter_list

def parse_meta(path):
    '''

    '''
    g = gzip.open(path, 'rb')
    inter_list = []
    for l in tqdm(g):
        inter_list.append(json.loads(l.decode()))

    return inter_list

def add_comma(num):
    '''

    1000000 -> 1,000,000
    '''
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []

    data_flie = './data/' + str(dataset_name) + '/raw/' + str(dataset_name) + '.json.gz'
    for inter in parse(data_flie):
        if float(inter['overall']) <= rating_score: 
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas

def Amazon2023(dataset_name, rating_score):
    '''
    '''
    datas = []
    data_flie = './data/' + str(dataset_name) + '/raw/' + str(dataset_name) + '.jsonl.gz'
    for inter in parse(data_flie):
        if float(inter['rating']) <= rating_score:
            continue
        user = inter['user_id']
        item = inter['parent_asin']
        time = inter['timestamp']
        datas.append((user, item, int(time)))
    return datas

def filter_common(user_items, user_t, item_t):
    '''

    '''
    user_count = defaultdict(int)
    item_count = defaultdict(int)

    for user, item, _ in user_items:
        user_count[user] += 1
        item_count[item] += 1
    # print(user_count['A3F5DIB5CVJAT0'])
    User = {}
    for user, item, timestamp in user_items:
        if user_count[user] < user_t or item_count[item] < item_t:
            continue
        if user not in User.keys():
            User[user] = []

        User[user].append((item, timestamp))


    new_User = {}
    for userid in User.keys():

        User[userid].sort(key=lambda x: x[1])

        new_hist = [i for i, t in User[userid]]

        new_User[userid] = new_hist

    return new_User

def id_map(user_items): # user_items dict
    '''

    '''

    user2id = {} 
    item2id = {} 
    id2user = {} 
    id2item = {} 

    user_id = 1

    item_id = 1

    final_data = {}

    for user, items in user_items.items():

        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1

        iids = [] # item id lists
        for item in items:

            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }

    return final_data, user_id-1, item_id-1, data_maps

def filter_minmum(user_items, min_len=3):
    '''

    '''

    new_user_items = {}
    for user, items in user_items.items():
        if len(items) >= min_len:
            new_user_items[user] = items

    return new_user_items

def get_counts(user_items):
    '''

    '''

    user_count = {}
    item_count = {}

    for user, items in user_items.items():
        user_count[user] = len(items)
        for item in items:
            if item not in item_count.keys():
                item_count[item] = 1
            else:
                item_count[item] += 1


    return user_count, item_count

def Amazon_meta(dataset_name, data_maps):
    '''

    asin - ID of the product, e.g. 0000031852
    --"asin": "0000031852",
    title - name of the product
    --"title": "Girls Ballet Tutu Zebra Hot Pink",
    description
    price - price in US dollars (at time of crawl)
    --"price": 3.17,
    imUrl - url of the product image (str)
    --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
    related - related products (also bought, also viewed, bought together, buy after viewing)
    --"related":{
        "also_bought": ["B00JHONN1S"],
        "also_viewed": ["B002BZX8Z6"],
        "bought_together": ["B002BZX8Z6"]
    },
    salesRank - sales rank information
    --"salesRank": {"Toys & Games": 211836}
    brand - brand name
    --"brand": "Coxlures",
    categories - list of categories the product belongs to
    --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    '''
    datas = {}
    meta_flie = './data/' + str(dataset_name) + '/raw/meta_' + str(dataset_name) + '.json.gz'
    item_asins = list(data_maps['item2id'].keys())
    # no_title_count = 0
    for info in tqdm(parse_meta(meta_flie)):

        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info

    # print(no_title_count)
    return datas

def Amazon_meta2023(dataset_name, data_maps):
    '''

    '''
    datas = {}
    meta_flie = './data/' + str(dataset_name) + '/raw/meta_' + str(dataset_name) + '.jsonl.gz'
    item_asins = list(data_maps['item2id'].keys())
    for info in tqdm(parse_meta(meta_flie)):

        if info['parent_asin'] not in item_asins:
            continue
        datas[info['parent_asin']] = info
    return datas

def get_attribute_Amazon(meta_infos, datamaps):
    '''

    '''

    attributes = defaultdict(int)

    attribute2id = {}

    id2attribute = {}

    attributeid2num = defaultdict(int)

    attribute_id = 1

    items2attributes = {}

    attribute_lens = []


    for iid, attributes in meta_infos.items():

        item_id = datamaps['item2id'][iid]

        items2attributes[item_id] = []

        for attribute in attributes:

            if attribute not in attribute2id:
  
                attribute2id[attribute] = attribute_id

                id2attribute[attribute_id] = attribute

                attribute_id += 1

            attributeid2num[attribute2id[attribute]] += 1

            items2attributes[item_id].append(attribute2id[attribute])

        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')

    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes

def Yelp(date_min, date_max, rating_score):
    '''

    take out inters in [date_min, date_max] and the score < rating_score
    '''
    datas = []
    data_flie = './data/yelp/raw/yelp_academic_dataset_review.json'
    lines = open(data_flie).readlines()
    for line in tqdm(lines):
        review = json.loads(line.strip())
        user = review['user_id']
        item = review['business_id']
        rating = review['stars']
        # 2004-10-12 10:13:32 2019-12-13 15:51:19
        date = review['date']
        if date < date_min or date > date_max or float(rating) <= rating_score:
            continue
        time = date.replace('-','').replace(':','').replace(' ','') 
        datas.append((user, item, int(time)))
    return datas

def Yelp_meta(datamaps):
    '''

    '''
    meta_infos = {}
    meta_file = './data/yelp/raw/yelp_academic_dataset_business.json'
    item_ids = list(datamaps['item2id'].keys())
    lines = open(meta_file).readlines()
    for line in tqdm(lines):
        info = json.loads(line)
        if info['business_id'] not in item_ids:
            continue
        meta_infos[info['business_id']] = info
    return meta_infos

def get_attribute_Yelp(meta_infos, datamaps, attribute_core):
    '''

    '''
    attributes = defaultdict(int)
    for iid, info in tqdm(meta_infos.items()):
        try:
            cates = [cate.strip() for cate in info['categories'].split(',')]
            for cate in cates:
                attributes[cate] +=1
        except:
            pass
    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm(meta_infos.items()):
        new_meta[iid] = []
        try:
            cates = [cate.strip() for cate in info['categories'].split(',') ]
            for cate in cates:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
        except:
            pass

    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'after delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')

    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes

def get_interaction(datas):
    '''
    sort the interactions based on timestamp
    '''
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():

        item_time.sort(key=lambda x: x[1])
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

def main(data_name, data_type='Amazon', user_core=3, item_core=5):
    assert data_type in {'Amazon','Yelp','Amazon2023'}
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted

    attribute_core = 0
    if data_type == "Amazon":
        datas = Amazon(data_name, rating_score=rating_score)
    elif data_type == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2000-01-01 00:00:00'
        datas = Yelp(date_min, date_max, rating_score)
    elif data_type == "Amazon2023":
        datas = Amazon2023(data_name, rating_score=rating_score)

    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')

    user_items = filter_common(datas, user_t=user_core, item_t=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = id_map(user_items)

    user_items = filter_minmum(user_items, min_len=3)

    user_count, item_count = get_counts(user_items)

    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)

    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)

    interact_num = np.sum([x for x in user_count_list])

    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)
    print(f"After filter User:{len(user_items)}\n")

    print('Begin extracting meta infos...')

    if data_type == 'Amazon':
        meta_infos = Amazon_meta(data_name, data_maps) 
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps)
    elif data_type == 'Amazon2023':
        meta_infos = Amazon_meta2023(data_name, data_maps) 
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps)
    elif data_type == 'Yelp':
        meta_infos = Yelp_meta(data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Yelp(meta_infos, data_maps, attribute_core)
    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
        f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
        f'{avg_attribute:.1f} \\')

    handled_path = 'data/' + data_name + '/handled/'
    if not os.path.exists(handled_path):
        os.makedirs(handled_path)

    data_file = handled_path + 'inter_seq.txt'
    item2attributes_file = handled_path + 'item2attributes.json'
    id_file = handled_path + "id_map.json"

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    json_str = json.dumps(meta_infos)

    with open(item2attributes_file, 'w') as out:
        out.write(json_str)

    with open(id_file, "w") as f:
        json.dump(data_maps, f)

if __name__ == "__main__":

    # main("beauty", data_type="Amazon", user_core=3, item_core=5)
    # main("fashion", data_type="Amazon", user_core=3, item_core=3)
    # main('yelp', data_type='Yelp', user_core=3, item_core=3)
    # main("beauty2023", data_type="Amazon2023", user_core=3, item_core=3)
    # main("Lbeauty", data_type="Amazon", user_core=3, item_core=5)
    # main("beautyPC", data_type="Amazon2023", user_core=3, item_core=3)
    # main("fashion2023", data_type="Amazon2023", user_core=3, item_core=3)
    # main("toys_and_games", data_type="Amazon", user_core=3, item_core=5)
    # main("video_games", data_type="Amazon", user_core=3, item_core=5)
    main("book", data_type="Amazon", user_core=3, item_core=5)