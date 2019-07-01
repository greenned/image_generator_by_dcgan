from bs4 import BeautifulSoup
import requests
from crawl_config import BASE_URL
import urllib.request
import ssl

class standardCralwer:
    def __init__(self, base_url=None):
        self.base_url = base_url

    def request_base_url(self):
        soup = None
        data = requests.get(self.base_url, verify=True).text
        soup = BeautifulSoup(data, 'html.parser')
        return soup

    def request_url(self, url):
        soup = None
        data = requests.get(url).text
        soup = BeautifulSoup(data, 'html.parser')
        return soup

    def soup_select(self, soup, select=None, cls=None):
        selected = None
        if cls is None:
            selected = soup.find_all(select)
        else:
            selected = soup.find_all(select, {"class":cls})
        return selected

        #TODO 이미지 접근해서 메타정보와 이미지 파일로 다운로드

    def parse_text(self, soup_lst):
        parsed = list()
        for soup in soup_lst:
            parsed.append(soup.text)
        return parsed
    
    def parse_item(self, soup_lst, item):
        parsed = list()
        for soup in soup_lst:
            parsed.append(soup.get(item))
        return parsed
    
    def get_image(self, img_url):
        img = None
        context = ssl._create_unverified_context()
        img = urllib.request.urlopen(img_url, context=context).read()
        return img




if __name__ == "__main__":
    pass
    # sc = standartCralwer("https://terms.naver.com/entry.nhn?docId=1553254&cid=46702&categoryId=46753")
    # soup = sc.request_base_url()
    # selected = sc.soup_select(soup, "img")

    # parsed = sc.parse_text(selected)
    # parsed_item = sc.parse_item(selected, "href")
    # print(selected[0].get("origin_src"))
    # sc.get_image(selected[0].get("origin_src"), "hi")
    # print(selected[0].find_all("td"))
