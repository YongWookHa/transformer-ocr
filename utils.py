import yaml
import pickle
from easydict import EasyDict

def load_setting(setting):
    with open(setting, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(cfg)

def save_tokenizer(tokenizer, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print("tokenizer saved in {}".format(save_path))
