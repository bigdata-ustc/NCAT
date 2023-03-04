import json
import random
import numpy as np

min_log = 10
min_exer = 50 
min_kc = 10 
min_cat_log = 100 

def preprocess(data_name):
    with open('data/%s/log_data.json' % data_name, encoding='utf8') as i_f:
        stus = json.load(i_f) # list
    filter_dump_stus = [] 
    for stu in stus:
        exer_cnt = {}
        for log in stu['logs']:    
            if log["exer_id"] not in exer_cnt:
                exer_cnt[log["exer_id"]] = {
                    'cnt':0,
                    1:0,
                    0:0
                }
            exer_cnt[log["exer_id"]]['cnt'] += 1
            exer_cnt[log["exer_id"]][log['score']] += 1
        new_logs = []
        for log in stu['logs']:  
            if log["exer_id"] in exer_cnt:
                if exer_cnt[log["exer_id"]][1] > exer_cnt[log["exer_id"]][0]:
                    log["score"] = 1
                else:
                    log["score"] = 0
                new_logs.append(log)
                del exer_cnt[log["exer_id"]]
        filter_dump_stus.append({"user_id":stu["user_id"], "log_num":len(new_logs), "logs":new_logs})
    stus = filter_dump_stus

    exer_stu = {}
    kc_exer = {}
    for stu in stus:
        for log in stu['logs']:
            if log["exer_id"] not in exer_stu:
                exer_stu[log["exer_id"]] = set()
            exer_stu[log["exer_id"]].add(stu['user_id'])
            for kc in log['knowledge_code']:
                if kc not in kc_exer:
                    kc_exer[kc] = set()
                kc_exer[kc].add(log["exer_id"])

    filter_stus = []
    cnt = 0
    exer_set = set() 
    for stu in stus:
        logs = [] 
        for log in stu['logs']:
            if len(exer_stu[log["exer_id"]]) >= min_exer:
                new_kc = []
                for kc in log['knowledge_code']:
                    if len(kc_exer[kc]) >= min_kc:
                        new_kc.append(kc)
                if len(new_kc) == 0:
                    continue
                log['knowledge_code'] = new_kc
                logs.append(log)
            else:
                cnt += 1
        # filter by length
        if len(set([log['exer_id'] for log in logs])) >= min_log:
            filter_stus.append({"user_id":stu["user_id"], "log_num":len(logs), "logs":logs})
            exer_set = exer_set.union(set([log['exer_id'] for log in logs]))
    
    item_id_map = dict(zip(exer_set, np.arange(1, len(exer_set) + 1).tolist()))
    new_stus = []
    log_all, exer_set, kc_set = 0, set(), set() 

    for stu in filter_stus:
        logs = [] 
        for log in stu['logs']:
            log['exer_id'] = item_id_map[log['exer_id']]
            logs.append(log)

        new_stus.append({"user_id":stu["user_id"], "log_num":len(logs), "logs":logs})
        log_all += len(set([log['exer_id'] for log in logs]))
        exer_set = exer_set.union(set([log['exer_id'] for log in logs]))
        for log in logs:
            kc_set = kc_set.union(set(log['knowledge_code']))

    
    print(cnt)
    stu_all = len(new_stus)
    exer_all = len(exer_set)
    exer_maxid = max(exer_set)
    stu_maxid = max(set([s['user_id'] for s in stus]))
    kc_maxid = max(kc_set)
    kc_all = len(kc_set)
    filtered_info = "data_name: {}\nstu_all: {}\nstu_maxid: {}\nexer_all: {}\nexer_maxid: {}\nkc_all: {}\nkc_maxid: {}\nlog_all: {}\navg_per_stu: {}\n".format(\
        data_name, stu_all, stu_maxid ,exer_all, exer_maxid  ,kc_all, kc_maxid, log_all, log_all/stu_all)
    with open('data/%s/info_filtered.yml' % data_name, 'w', encoding='utf8') as output_file:
        print(filtered_info)
        output_file.write(filtered_info)

    with open('data/%s/log_data_filtered.json' % data_name, 'w', encoding='utf8') as output_file:
        json.dump(new_stus, output_file, indent=4, ensure_ascii=False)

def divide_data(data_name):
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set, val_set and test_set (0.7:0.1:0.2)
    :return:
    '''
    with open('data/%s/log_data_filtered.json' % data_name, encoding='utf8') as i_f:
        stus = json.load(i_f) # list
    
    # 1. delete students who have fewer than min_log response logs
    stu_len = len(stus) # stu num
    random.shuffle(stus)

    # 2. divide dataset into train_set, val_set and test_set
    train_set, val_set = [], []
    stus_test = [s for s in stus if len(s['logs']) >= min_cat_log] # 留做cat
    stus_train = stus
    # stus_train = [s for s in stus if len(s['logs']) < min_cat_log] # 留做cat
    print(f'all stu len: {stu_len} test_ids len: {len(stus_test)} cat/all_ratio: {len(stus_test) / stu_len} ')

    all_set = {}

    for stu in stus_train:
        user_id = stu['user_id']
        # shuffle logs in train_slice together, get train_set
        for log in stu['logs']:
            if log['exer_id'] not in all_set:
                all_set[log['exer_id']] = []
            all_set[log['exer_id']].append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                            'knowledge_code': log['knowledge_code']})
    for eid in all_set:
        random.shuffle(all_set[eid])
        train_size = int(len(all_set[eid]) * 0.8)
        train_set.extend(all_set[eid][:train_size])
        val_set.extend(all_set[eid][train_size:])

    random.shuffle(train_set)
    random.shuffle(val_set)
    print(f'train records: {len(train_set)}, val records: {len(val_set)}')

    with open('data/%s/train_set.json'% data_name, 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('data/%s/val_set.json' % data_name, 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)

    with open('data/%s/test_set.json' % data_name, 'w', encoding='utf8') as output_file:
        json.dump(stus_test, output_file, indent=4, ensure_ascii=False)





if __name__ == '__main__':
    # divide_data('assist0910')
    dataset_name = 'assist1213' # 'nips_edu' # assist1213
    print(dataset_name)
    preprocess(dataset_name)
    divide_data(dataset_name)


