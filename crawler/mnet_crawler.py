import requests, pickle, os, time
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import urllib.request

FOLDER_PATH = "/home/ysmetal/Dropbox/workspace/KANDINSKY/data"
PASS_CHAR  = ["/","\",""?""", "%", "*", ":", "|", """\"""", "<", ">", ".","(",")"]

class AlbumCrawler:
    def __init__(self):
        self.base_url = "http://www.mnet.com/album/{num}" # YYYYMMDD
        self.base_start_date = datetime(2011, 1, 1)
        self.base_end_date = datetime.now()
    
    def datetime_converter(self, date_time):
        return datetime.strftime(date_time, "%Y%m%d")

    def kpop_checker(self, soup):
        check = False
        categories = soup.find_all("span",{"class":"right"})
        for category in categories:
            if "가요" in category.text:
                check = True
                return check
        return check
    
    def get_data_from_reqeusts(self, url):
        soup = None
        data = requests.get(url)
        soup = BeautifulSoup(data.content, 'html.parser' , from_encoding="utf8") # 
        return soup

    def get_artist_name(self):
        pass
    
    def get_songs_name(self, soup):
        songs = list()
        song_class = soup.find_all('a',{"class":"MMLI_Song"})
        #print(song_class)
        for song in song_class:
            #print(song.text)
            #print(song.find("a"))
            songs.append(song.text)
        #print(songs)
        return songs
    
    def get_album_name(self, soup):
        album = None
        album = soup.find("p",{"class":"ml0"}).text
        album = self.replace_char(album).rstrip()
        return album
    
    def save_cover_image(self, soup, album_name):
        #album_name = None #self.replace_char(album_name)
        img_url = None
        img_url = soup.find("img").get("src")
        #print(img_url)
        urllib.request.urlretrieve(img_url, os.path.join(FOLDER_PATH, album_name, "{}.jpg".format(album_name)))
        return img_url
    
    def make_dir(self, album_name):
        os.makedirs(os.path.join(FOLDER_PATH, album_name))
        pass
    
    def make_txt(self, album_name, songs):
        #album_name = None #self.replace_char(album_name)
        with open("{}/{}/{}.txt".format(FOLDER_PATH, album_name, album_name), "w") as f:
            for song in songs:
                f.write(song + "\n")
        f.close()
        pass
    
    def replace_char(self, name):
        for char in PASS_CHAR:
            name = name.replace(char, "")
        return name
    
    def exist_checker(self, name):
        check = False
        albums = os.listdir(FOLDER_PATH)
        if name in albums:
            check = True

        return check

    def delete_dir(self, album_name):
        os.rmdir(os.path.join(FOLDER_PATH, album_name))
        

def main(debug=True):
    c = AlbumCrawler()
    for i in tqdm(range(808,3050000)):
        soup = c.get_data_from_reqeusts(c.base_url.format(num=i))
        if not c.kpop_checker(soup):
            continue
        album_name = c.get_album_name(soup)
        if c.exist_checker(album_name):
            continue
        try:
            songs = c.get_songs_name(soup)
            c.make_dir(album_name)
            c.make_txt(album_name, songs)
            c.save_cover_image(soup, album_name)
            # c.append_songs(album_name, songs)
        except (FileNotFoundError, urllib.error.HTTPError):
            try:
                c.delete_dir(album_name)
            except:
                pass
            continue

        if i % 500 == 0 and i != 0:
            time.sleep(3)
            print("{}까지 크롤링...".format(i))

if __name__ == "__main__":
    #print(PASS_CHAR)
    main()
    # a = AlbumCrawler()
    # soup = a.get_data_from_reqeusts(a.base_url.format(num=3270530))
    # print(a.kpop_checker(soup))
    # songs = a.get_songs_name(soup)
    # a.make_txt("hi", songs)
    #print(a.save_cover_image(soup,"hi"))
    #a.make_dir("hi")
    #print(soup)
    #print(a.get_album_name(soup))
    


    # data = a.get_top_50(a.base_url.format(date=a.datetime_converter(a.base_end_date)))
    # print(data.content)
    
