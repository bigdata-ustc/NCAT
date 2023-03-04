import os
import time

pwd_path = 'xxx/NCAT/data/assist1213/'

class Config(object):
    train_dataset_path = os.path.join(pwd_path + "2012-2013-data-with-predictions-4-final.csv")
    
    new_data_save_path = "./log_data.json"
    map_path = './map/'
