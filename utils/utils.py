import re
from kd_configs import FILE_REG
import shutil
from dbmanager import dbManager
from crawl_config import *

def parse_reg(regex, sentence):
    reg = re.compile(regex)
    item = reg.findall(sentence)
    return item

def file_mover(from_dir, to_dir, file_name):
    shutil.move(from_dir+file_name+"png", to_dir+file_name+"png")
    pass

if __name__ == "__main__":
    d = dbManager(HOST, PORT, DB, COLLECTION_NAME)
    item_dict = d.get_pair("_id","content")
    
    print(item_dict)


    sen = "/home/ysmetal/google-drive/workspace/KANDINSKY/data/img/1406307.png"
    item = parse_reg(FILE_REG, sen)
    print(item)