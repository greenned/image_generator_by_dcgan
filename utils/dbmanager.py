from pymongo import MongoClient

class dbManager:
    def __init__(self, host=None, port=None, db=None, collection =None):
        self.host = host
        self.port = port
        self.db = db
        self.collection_name = collection
        self.collection = self.get_collection()
    
    def get_collection(self):
        client = MongoClient(self.host, self.port)
        db = client[self.db]
        collection = db[self.collection_name]
        return collection

    def insert_data(self, data=None):
        if data is not None:
            self.collection.insert_one(data)
        else:
            return None
        return data
    
    def get_exist_item(self, key_name=None):
        item_list = list()
        for item in self.collection.find({}):
            target = item[key_name]
            if target is not None:
                item_list.append(target)
            else:
                continue
        return item_list
    
    def get_pair(self, key_name=None, value_name=None):
        pair_dict = dict()
        for item in self.collection.find({}):
            key = item[key_name]
            value = item[value_name]
            if key is None or value is None:
                continue
            else:
                pair_dict[key] = value
        return pair_dict
