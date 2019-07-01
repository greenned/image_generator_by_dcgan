from dbmanager import dbManager
from konlpy.tag import Okt
from crawl_config import *
from kd_configs import *
from gensim.models import Word2Vec
from tqdm import tqdm


class bagOfWord(dbManager):
    def __init__(self, value_name):
        super().__init__(HOST, PORT, DB, COLLECTION_NAME)
        self.contents = self.get_exist_item(value_name)
        # self.contents_dict = self.get_pair(key_name, value_name)
    
    def get_morph(self, sentence_in_list):
        if sentence_in_list is None:
            raise IOError
        morph_list = list()
        okt = Okt()
        for con in tqdm(sentence_in_list, total=len(sentence_in_list)):
            morph_list.append(okt.nouns(con))
        return morph_list

class w2vTrainer(bagOfWord):
    def __init__(self, value_name):
        super().__init__(value_name)
        self.bow = self.get_morph(self.contents)
    
    def train_w2v(self, bow, min_count, window, size, epoch):
        model = Word2Vec(bow, min_count=min_count, sg=1, window=window, size=size, iter=epoch)
        print("trained..")
        return model
    
    def save_model(self, model, filepath, filename):
        model.save("{}/{}.model".format(filepath, filename))
        print("Saved...")


if __name__ == "__main__":
    wt = w2vTrainer(VALUE_NAME)
    model = wt.train_w2v(wt.bow, min_count, window, size, epoch)
    wt.save_model(model, MODEL_PATH, MODEL_NAME)