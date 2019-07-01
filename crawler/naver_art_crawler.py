import requests, re, os, time
from crawl_config import *
from bs4 import BeautifulSoup
from tqdm import tqdm
from standard_crawler import standardCralwer
from dbmanager import dbManager

class fileManger:
    file_idx = 0
    pass

class artCrawler(dbManager, standardCralwer):
    def __init__(self, target_info = TARGET_INFO):
        dbManager.__init__(self, HOST, PORT, DB, COLLECTION_NAME)
        standardCralwer.__init__(self, BASE_URL)
        self.target_info = target_info
        self.info_dict = self.initialize_dict()
        self.pk_reg = re.compile(PK_REG)
        self.filepath = FILEPATH
    
    # 인서트할 딕셔너리 초기화
    def initialize_dict(self):
        info_dict = dict()
        for info in self.target_info:
            info_dict[info] = None
        return info_dict

    def get_category_info(self):
        info_dict = dict()
        soup = self.request_base_url()
        selected = self.soup_select(soup, "a", "title")
        for sel in selected:
            info_dict[sel.text] = CATEGORY_URL + sel.get("href")
        return info_dict
    
    def get_art_url(self, soup):
        url_list = list()
        datas = self.soup_select(soup, "strong", "title")
        for data in datas:
            try:
                # print(data.find("a").get("href"))
                url = data.find("a").get("href")
                tf = url.find("entry")
                if tf == 1:
                    url_list.append(CATEGORY_URL+url)
                else:
                    continue
            except AttributeError:
                continue

        return url_list
    
    def make_page_url(self, category_url, num):
        return str(category_url + "&page=" + str(num))

    def crawl_art_info(self, art_url, category):
        name = None
        pk = None
        content = None
        filename = None
        info_dict = self.info_dict.copy()
        result = [None, None, None]
        soup = self.request_url(art_url)
        try:
            name = self.soup_select(soup,"h2","headword")[0].text.strip()
            pk = self.pk_reg.findall(art_url)[0]
            content = self.soup_select(soup, "p", "txt")
            info_dict['title'] = name
            info_dict['_id'] = pk
            try:
                info_dict['content'] = content[0].text
                info_dict['is_content'] = True
            except:
                info_dict['is_content'] = False
            info_dict['is_error'] = False
        except:
            info_dict['is_error'] = True
        filename = os.path.join(self.filepath, pk + ".png") #+ "__" + name
        img_url = self.soup_select(soup, "img")[0].get("origin_src")
        info_dict["img_url"] = img_url
        info_dict["art_url"] = art_url
        info_dict["category"] = category
        img = self.get_image(img_url)
        return filename, info_dict, img
    
    def save_data(self, filename, info_dict, img):
        # 이미지 저장하기
        with open(filename, mode="wb") as f:
            f.write(img)
        # DB저장하기
        self.insert_data(info_dict)
        return None
    
    def exist_check(self, art_url_list, exist_art_url):
        check = False
        for art_url in art_url_list:
            if art_url in exist_art_url:
                check = True
                return check
            else:
                continue
        return check

def main(debug=False):
    ac = artCrawler()
    # 카테고리 수집
    category_dict = ac.get_category_info()
    
    for category, url in tqdm(category_dict.items(),total=len(category_dict.keys())):
        # 페이지 URL
        for i in tqdm(range(2000)):
            page_url = ac.make_page_url(url, i)
            # 그림 URL따기
            soup = ac.request_url(page_url)
            art_url_list = ac.get_art_url(soup)
            exist_art_url = ac.get_exist_item("page_url")
            if ac.exist_check(art_url_list, exist_art_url):
                break
            # 데이터 크롤
            for art_url in art_url_list:
                if art_url not in exist_art_url:
                    filename, info_dict, img = ac.crawl_art_info(art_url, category)
                    #if debug: print(info_dict)
                    # DB 및 파일저장
                    try:
                        ac.save_data(filename, info_dict, img)
                    except:
                        print(filename)
                        if debug: raise
                        continue
                else:
                    continue
            time.sleep(SLEEP_TIME)
    return None


if __name__ == "__main__":
    main()
    # data = {"hi":1, "hello":2}
    # artCrawler()


