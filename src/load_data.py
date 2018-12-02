
"""
from sklearn.datasets import load_files

categories = ['bbcsport.rugby, bbcsport.cricket']
data_train = load_files(container_path='../data/data_Sets/bbcsport', decode_error='ignore', encoding='utf-8')


print(data_train.target_names)
"""
"""

import os

SKL_DATA = "SCIKIT_LEARN_DATA"
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "fixtures")


def get_data_home(data_home=None):

    ###Returns the path of the data directory

    if data_home is None:
        data_home = os.environ.get(SKL_DATA, DATA_DIR)

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home
"""
"""
import os

import sklearn
from sklearn import datasets


DATA_SETS_DIR = '../data/data_sets'
DATA_SET_NAME = 'bbcsport'
CATEGORIES = ['rugby, cricket'] #['bbcsport.rugby, bbcsport.cricket']
DATA_SET_CONTAINER_PATH = os.path.join(DATA_SETS_DIR, DATA_SET_NAME) 
#TRAIN_DIR = '../data/train'

print(DATA_SET_CONTAINER_PATH)

data_train = datasets.load_files(container_path=DATA_SET_CONTAINER_PATH, \
    description=None, \
	categories=CATEGORIES, load_content=True, shuffle=True, \
	encoding='utf-8', decode_error='ignore', random_state=0)

print(data_train.target_names)
"""

from sklearn.datasets import load_files

container_path = '../data/data_sets/bbcsport'

data_train = load_files(container_path=container_path, decode_error='ignore', encoding='utf-8')


print(data_train.target_names)