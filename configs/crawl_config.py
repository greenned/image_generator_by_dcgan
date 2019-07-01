
# DB정보
HOST = "greenned.iptime.org"
PORT = 27017
DB = "kandinsky"
COLLECTION_NAME = "img_meta"

# 사이트 정보
BASE_URL = "https://terms.naver.com/list.nhn?cid=46702&categoryId=46741"
CATEGORY_URL = "https://terms.naver.com"
#TARGET_INFO = ["name", "title_kor", "title_eng", "year", "trend", "type", "content"]
TARGET_INFO = ["_id", "title", "content", "img_url","page_url"]

# 화가이름, 작품명(한글), 작품명(영어), 이미지, 제작연도, 사조, 종류, 설명

# 파일정보
FILEPATH = "./data/img"

# 수집관련
SLEEP_TIME = 2
PK_REG = "(?<=docId=)\d+"