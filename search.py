import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
target_text = """Pfizer vaccine "page 132" warns not to have unprotected sex for 28 days after the second dose of the COVID-19 vaccine because "genetic manipulation" may cause birth defects.""" 
def keyword_tfidf_analysis(target, candidates):
    # 提取目标文本关键词
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
    target_vec = vectorizer.fit_transform([target])
    feature_names = vectorizer.get_feature_names_out()
#     print(f"关键特征词: {feature_names}")

    # 计算每个候选文档的匹配度
#     candidate_texts = [c["content"] for c in candidates]
    candidate_vecs = vectorizer.transform([candidates])
    
    # 计算余弦相似度
    scores = (candidate_vecs * target_vec.T).toarray().flatten()
    return scores
from datetime import datetime

def parse_custom_date(date_str):
    """解析包含'Published'前缀的日期"""
    try:
        # 去除前缀并去除前后空格
        clean_str = date_str.split("Published")[-1].strip()
        return datetime.strptime(clean_str, "%b %d, %Y")
    except ValueError as e:
        raise ValueError(f"日期解析失败: {date_str}") from e

def calculate_day_diff(date_str1,date_str2):

    # 解析第一个日期
    dt1 = parse_custom_date(date_str1)
    
    # 解析ISO 8601日期（带时区）
    dt2 = datetime.fromisoformat(date_str2.replace("Z", "+00:00"))  # 转换为UTC时间
    
    # 统一时区（将第一个日期设为UTC）
    dt1_utc = dt1.astimezone(dt2.tzinfo)
    
    # 计算天数差（基于日期部分）
    days_diff = (dt2.date() - dt1_utc.date()).days
    
    return days_diff

# 执行计算
try:
    date_str1 = "Published May 9, 2019"
    date_str2 = "2025-03-16T14:11:18.531Z"
    difference = calculate_day_diff(date_str1,date_str2)
    print(f"天数差：{difference} 天")
except Exception as e:
    print(f"错误：{str(e)}")
import requests
from bs4 import BeautifulSoup
from readability import Document
import os
import spacy
from bs4 import BeautifulSoup
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'  # set the http proxy information
# 1. 请求网页
def get_text_time(url):
#     url = "https://www.snopes.com/fact-check/covid-vaccine-unprotected-sex/"  # 替换为你要抓取的网页链接
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding  # 设置正确的编码

    # 2. 使用 readability 提取主要内容
    doc = Document(response.text)
    title = doc.title()
    summary_html = doc.summary()

    # 3. 用 BeautifulSoup 解析提取出的 HTML 内容
    soup = BeautifulSoup(summary_html, 'html.parser')
    text = soup.get_text(separator="\n", strip=True)

    # 打印结果
    # print("标题：", title)
    # print("正文：\n", text)
    soup = BeautifulSoup(response.text, "html.parser")
    meta_time = soup.find("meta", {"property": "article:published_time"})
    if meta_time and meta_time.has_attr("content"):
        return text,meta_time["content"]
    else:
        return text,None
def get_text_only(url):
#     url = "https://www.snopes.com/fact-check/covid-vaccine-unprotected-sex/"  # 替换为你要抓取的网页链接
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding  # 设置正确的编码

    # 2. 使用 readability 提取主要内容
    doc = Document(response.text)
    title = doc.title()
    summary_html = doc.summary()

    # 3. 用 BeautifulSoup 解析提取出的 HTML 内容
    soup = BeautifulSoup(summary_html, 'html.parser')
    text = soup.get_text(separator="\n", strip=True)
    return text
url="""https://www.snopes.com/fact-check/marathon-cigarettes-1984-olympics/"""
text1,time1=get_text_time(url)
import re
from nltk import sent_tokenize
import spacy

# 初始化 spaCy 模型
nlp = spacy.load("en_core_web_sm")

def clean_text(text, lowercase=False):
    """预处理文本：清理特殊字符，可选是否转为小写"""
    text = re.sub(r'\n', ' ', text)                   # 去除换行符
    text = re.sub(r'\s+', ' ', text)                  # 合并多个空格
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)      # 移除非常规符号（保留句号和逗号）
    text = text.strip()
    return text.lower() if lowercase else text  # 按需转为小写

def extract_entities(text):
    """使用 spaCy 提取命名实体（保留原始大小写）"""
    doc = nlp(text)
    entities = {
        "PERSON": [],
        "ORG": [],
        "DATE": [],
        "GPE": [],      # 地理政治实体（如城市、国家）
        "EVENT": [],
        "NORP": []      # 民族/宗教/政治团体（如 Caucasian, Hispanic）
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            # 保留原始文本，后续匹配时再统一小写
            entities[ent.label_].append(ent.text)  
    return entities
def get_entity(text):
    cleaned_statement = clean_text(text, lowercase=False)
    stmt_entities = extract_entities(cleaned_statement)
    print(stmt_entities)
    # 合并关键实体（匹配时统一小写）
    keywords = {
        "person": [e.lower() for e in stmt_entities["PERSON"]],
        "location": [e.lower() for e in stmt_entities["GPE"]],
        "date": [e.lower() for e in stmt_entities["DATE"]],
        "race": [e.lower() for e in stmt_entities["NORP"]],
        "event": [e.lower() for e in stmt_entities["EVENT"]],  # 补充隐含实体
        "ORG": [e.lower() for e in stmt_entities["ORG"]],
    }
    return keywords
def find_context(document, keywords, window=2):
    """基于实体匹配的上下文提取"""
    # 预处理文本
    cleaned_doc = clean_text(document, lowercase=False)
    cleaned_doc_lower = cleaned_doc.lower()
    sentences = sent_tokenize(cleaned_doc_lower)
    # 匹配逻辑
    max_score=0
    for i, sentence in enumerate(sentences):
        score = 0
        # 检查人物匹配（Kyle Rittenhouse）
        if any(person in sentence for person in keywords["person"]):
            score += 2  # 人物权重更高
        # 检查地点（Wisconsin, Kenosha）
        if any(loc in sentence for loc in keywords["location"]):
            score += 1
        # 检查时间（2020）
        if any(date in sentence for date in keywords["date"]):
            score += 1
        # 检查事件（protest, court case）
        if any(event in sentence for event in keywords["event"]):
            score += 1
        if any(org in sentence for org in keywords["ORG"]):
            score += 1
        if score>max_score:
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            max_score=score
            s_window=' '.join(sentences[start:end])
        # 若总评分达到阈值，返回上下文窗口
        if score >= 3:
#             start = max(0, i - window)
#             end = min(len(sentences), i + window + 1)
            return ' '.join(sentences[start:end])
    if max_score>0:
        return s_window
    else:
        return ''

# 输入数据
original_statement = """A real vintage advertisement for Camels featured a doctor and the slogan "More doctors smoke Camels than any other cigarette."""

document = text1

# 执行并打印结果
keyword1=get_entity(original_statement)
context = find_context(document, keyword1)
print("匹配的上下文：\n", context)
# 图片搜索
import base64

def save_base64_image(data_uri, save_path):
    try:
        # 检查是否是 Base64 数据URI
        if not data_uri.startswith("data:image/"):
            raise ValueError("不是有效的图片数据URI")
            
        # 分割头部和数据部分
        header, data = data_uri.split(",", 1)
        
        # 提取 MIME 类型和编码方式
        mime_type = header.split(":")[1].split(";")[0]
        encoding = header.split(";")[1]
        
        if encoding != "base64":
            raise ValueError("仅支持 Base64 编码")
        
        # 解码 Base64 数据
        image_data = base64.b64decode(data)
        
        # 保存文件
        with open(save_path, "wb") as f:
            f.write(image_data)
            
        print(f"图片已保存至 {save_path}")
        return True
        
    except Exception as e:
        print(f"下载失败: {str(e)}")
        return False

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options as EdgeOptions
import os
import requests
from bs4 import BeautifulSoup
import os
import re
from tqdm import tqdm 
from urllib.parse import urljoin
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'  # set the http proxy information

# 本地图片路径
image_path =r"E:\finefake\FineFake\Image\snope\7.jpeg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"图片文件不存在: {image_path}")

# 初始化 Chrome WebDriver（确保 chromedriver 已加入系统 PATH）
service = Service(executable_path=r"E:\Downloads\edgedriver_win64\msedgedriver.exe")  # Replace with your ChromeDriver path
options = webdriver.ChromeOptions()
options = EdgeOptions()

# 关键反检测配置 --------------------------
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--lang=zh-CN")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0")
pattern = r'",\s*"(https?://[^"]+)"'
pattern2 = r'",\s*"https?://[^"]+","(.*?)"'
pattern3 = r'src="(data:image/jpeg;[^"]+)"/></div></div></div></div></div>'

# df['url']=''
# 禁用无头模式（如需必须无头，见后续补充方案）
# options.add_argument("--headless")

# 设置代理（推荐使用住宅代理）
# options.add_argument("--proxy-server=http://user:pass@host:port")

# 初始化驱动
service = webdriver.edge.service.Service(executable_path=r"E:\Downloads\edgedriver_win64\msedgedriver.exe")
driver = webdriver.Edge(service=service, options=options)
def scrape_google_search(url):
    try:
        driver.get(url)
        
        # 等待主要内容加载
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div#search")))
        
        # 模拟人类滚动加载更多内容
#         last_height = driver.execute_script("return document.body.scrollHeight")
#         for _ in range(3):  # 滚动3次加载更多结果
#             driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#             time.sleep(random.uniform(2.5, 4.0))
#             new_height = driver.execute_script("return document.body.scrollHeight")
#             if new_height == last_height:
#                 break
#             last_height = new_height
        
        # 定位图片和链接元素
        results = []
        items = driver.find_elements(By.CSS_SELECTOR, "div.g")
        print(driver.page_source)
        for item in items:
            try:
                # 提取图片链接
                img = item.find_element(By.CSS_SELECTOR, "img[src^='http']")
                img_url = img.get_attribute("src")
                
                # 提取目标网站链接
                link = item.find_element(By.CSS_SELECTOR, "a[href]")
                target_url = link.get_attribute("href")
                
                results.append({
                    "image": img_url,
                    "url": target_url
                })
            except:
                continue
        
        return results
    
    except Exception as e:
        print(f"抓取失败: {e}")
    finally:
#         driver.quit()
        pass
def get_img_evidence(image_path):
    driver.get("https://www.google.com.hk/imghp?hl=en&ogbl")
    time.sleep(2)  # 等待页面加载

    # 点击“按图片搜索”按钮（点击相机图标）
    # 注意：不同地区的 Yandex 页面可能有不同的元素定位方式，
    # 这里采用 aria-label 属性匹配，如果不行可尝试调整定位策略
    camera_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "div[aria-label='Search by image']"))
    )
    camera_btn.click()
        
    time.sleep(1)

    # 定位上传文件的 input 元素并上传图片
    file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
    file_input.send_keys(image_path)
    
    # 等待搜索结果加载
    time.sleep(1)
    
    # 输出当前页面 URL，即包含相似图片结果的页面
    current_url = driver.current_url
    print("搜索图片结果页面 URL：", current_url)
#     print(driver.page_source)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    link_image_list = []
#     img_tag = soup.find_all('img', class_="VeBrne")
    text=re.findall(pattern,str(soup))

    head=re.findall(pattern2, str(soup))
    img_tag=re.findall(pattern3, str(soup))
#     print(img_tag)
    a=min(len(text),len(head),len(img_tag),5)
    print('图片搜索条数:',a)
    for i in range(a):
#         print('i:',i)
#         print(len(img_tag))
        link_image_list.append((head[i],text[i],img_tag[i]))
    for head1,link, img in link_image_list:
        print("搜索图片标题:",head1)
        print("搜索图片网页链接:", link)
#         print("图片链接:", img)
#         print("-" * 60)
    return head[:a],text[:a],img_tag[:a]
# text1
# In 2021, Guinness World Records named paint developed by researchers at Purdue University the world's whitest.
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import os
import spacy
from bs4 import BeautifulSoup

os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'  # set the http proxy information

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
final_sentences=[]
def search_factcheck(query,index):
    # Set up Selenium with a Chrome driver
    service = Service(executable_path=r"E:\Downloads\edgedriver_win64\msedgedriver.exe")  # Replace with your ChromeDriver path
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (no GUI)
    driver = webdriver.Edge(service=service)  

    # try:
#     driver.get(url)

#     soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 构建搜索 URL
    search_url = f"https://toolbox.google.com/factcheck/explorer/search/{query};hl=en"
    driver.get(search_url)

    # 等待页面加载
    time.sleep(3)

    # 获取渲染后的 HTML
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # 查找搜索结果（这里结构可能变化，需实际调试）
    cards = soup.find_all("div", class_="fc-content-card")

    print(f"知识核查网站发现{len(cards)}条结果")
    if not os.path.exists('./knowledge/'+str(index)):
        os.makedirs('./knowledge/'+str(index))
    index1=0
    claim1=[]
    for card in cards:
#         print(card)
        title = card.find(class_="fc-claimer-name")
        publisher = card.find(class_="fc-review-publisher")
        claim = card.find(class_="fc-claim")
#         verdict = card.find("div", class_="fc-rating")
        link = card.find("a", href=True)

        print("原声明:", claim.text.strip() if claim else "N/A")
#         print("✅ Verdict:", verdict.text.strip() if verdict else "N/A")
        print("判断:", publisher.text.strip() if publisher else "N/A")
        print("链接:", link['href'] if link else "N/A")
        with open('./knowledge/'+str(index)+'/'+str(index1)+'.txt', 'w', encoding='utf-8') as f:
            data=[claim.text.strip(),publisher.text.strip(),link['href']]
            f.write(f"({'，'.join(data)})") 
        index1=index1+1
        claim1.append(f"({'，'.join(data)})")
    list1=f"({'，'.join(claim1)})"
    return list1
        
#         print(index1)
#         print("-" * 60)

#     driver.quit()


import re
import json
from urllib.parse import quote
from lxml import etree
import requests
import os
import spacy
from tqdm import tqdm 
from bs4 import BeautifulSoup
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'  # set the http proxy information
import feedparser
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from urllib.parse import quote
def get_text_url(text):
    def fetch_google_news_rss(query):
        """
        解析 Google News RSS 订阅源。
        :param query: 搜索关键词
        :return: 新闻列表
        """
        # 对查询参数进行 URL 编码
        encoded_query = quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}"
        feed = feedparser.parse(url)
        return feed.entries
    start_time = time.time()
    # 示例用法
    query = text
#     print(query)
    news_results = fetch_google_news_rss(query)
    link=[]
    for news in news_results:
#         print(f"标题: {news['title']}")
#         print(f"链接: {news['link']}")
#         print(f"发布时间: {news['published']}")
#         print("---")
        link.append(news['link'])
    if link==None:
        return None,None
    def get_google_params(url):
        """
        从给定的Google新闻链接中获取所需的参数：source、sign 和 ts
        """
        proxies = {
            'http': 'http://127.0.0.1:7897',
            'https': 'http://127.0.0.1:7897'
        }
        # 发送GET请求，获取HTML内容
        response = requests.get(url, proxies=proxies)
        # 使用lxml解析HTML
        tree = etree.HTML(response.text)
        # 使用XPath提取参数
        sign = tree.xpath('//c-wiz/div/@data-n-a-sg')[0]
        ts = tree.xpath('//c-wiz/div/@data-n-a-ts')[0]
        source = tree.xpath('//c-wiz/div/@data-n-a-id')[0]
        return source, sign, ts

    def get_origin_url(source, sign, ts):
        """
        根据提取的参数构造请求并获取重定向的原始新闻链接
        """
        url = f"https://news.google.com/_/DotsSplashUi/data/batchexecute"
        req_data = [[[  
            "Fbv4je",  # 请求类型
            f"[\"garturlreq\",[[\"zh-HK\",\"HK\",[\"FINANCE_TOP_INDICES\",\"WEB_TEST_1_0_0\"],null,null,1,1,\"HK:zh-Hant\",null,480,null,null,null,null,null,0,5],\"zh-HK\",\"HK\",1,[2,4,8],1,1,null,0,0,null,0],\"{source}\",{ts},\"{sign}\"]",
            None,
            "generic"
        ]]]
        payload = f"f.req={quote(json.dumps(req_data))}"
        headers = {
          'Host': 'news.google.com',
          'X-Same-Domain': '1',
          'Accept-Language': 'zh-CN',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0',
          'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
          'Accept': '*/*',
          'Origin': 'https://news.google.com',
          'Referer': 'https://news.google.com/',
          'Accept-Encoding': 'gzip, deflate, br',
        }
        proxies = {
            'http': 'http://127.0.0.1:7897',
            'https': 'http://127.0.0.1:7897'
        }
        # 发送POST请求，获取响应
        response = requests.post(url, headers=headers, data=payload, proxies=proxies)
        # 使用正则表达式匹配重定向URL
        match = re.search(r'https?://[^\s",\\]+', response.text)
        if match:
            redirect_url = match.group()
    #         print(redirect_url)
            return redirect_url

    res_url=[]
    print("文本搜索的重定向链接")
    p=0
    for rss_url in link:
        source, sign, ts = get_google_params(rss_url)
        redirect_url=get_origin_url(source, sign, ts)
        res_url.append(redirect_url)
        p=p+1
#         print(redirect_url)
        if p==5:
            break
    return res_url

# 用法示例

import pandas as pd
import time
df=pd.read_csv('snope1512_1020.csv')

for index, row in tqdm(df.iterrows()):
    if index<1019:
        continue
    print(f'第{index}条数据')
#     if not os.path.exists('./search/'+str(index)):
#         os.makedirs('./search/'+str(index))
    text=row['text']
    start_time0 = time.time()
    print('开始知识库搜索'+"-" * 60)
    try:
        list1=search_factcheck(text,index)
        df.at[index, 'knowledge_search']=list1
    except:
        pass
    start_time1 = time.time()
    print('用时：',start_time1-start_time0)
    print('开始文本搜索'+"-" * 60)
    if not os.path.exists('./text_evidence/'+str(index)):
        os.makedirs('./text_evidence/'+str(index))
    if not os.path.exists('./raw_text/'+str(index)):
        os.makedirs('./raw_text/'+str(index))
    try:
        link=get_text_url(text)
        keywords11=get_entity(text)
        index11=0
        text_evi=[]
        for link1 in link:
            print(link1)
            sea_text,time1=get_text_time(link1)
            if not os.path.exists('./text_evidence/'+str(index)+'/'+str(index11)):
                os.makedirs('./text_evidence/'+str(index)+'/'+str(index11))
            with open('./raw_text/'+str(index)+'/'+str(index11)+'.txt', 'w', encoding='utf-8') as f:
                f.write(sea_text) 
            tf=keyword_tfidf_analysis(text, sea_text)
            print('tf-idf得分:',tf)
    #         No matching context found.
            context=find_context(sea_text,keywords11)
            if context=='':
                print("空")
            print('文本相关上下文:',context)
    #         print('1',time1)
            if time1!=None:
                dif=calculate_day_diff(row['date'],time1)
                print(f'目标网页与源数据天数差{dif}天')
            with open('./text_evidence/'+str(index)+'/'+str(index11)+'.txt', 'w', encoding='utf-8') as f:
                data=[str(index11),link1,str(tf),context,str(dif)]
                f.write(f"({'，'.join(data)})") 
            if context!='':
    #             print('保存中')
                data1=[str(index11),link1,str(tf),context,str(dif)]
                text_evi.append(f"({'，'.join(data1)})")
            index11=index11+1
        df.at[index, 'text_evidence'] = f"({'。'.join(text_evi)})"
        print(f"({'。'.join(text_evi)})")
    except:
        pass
            
#         print(row['date'])
#         print(time1)
#         print(f'目标网页与源数据天数差{dif}天')
    start_time2 = time.time()
    print('用时：',start_time2-start_time1)
    image_path='E:/finefake/FineFake/'+str(row['image_path'])
    print('开始图片搜索'+"-" * 60)
    img_evi=[]
    try:
        head,link,img_tag1=get_img_evidence(image_path)
        if not os.path.exists('./img_evidence/'+str(index)):
            os.makedirs('./img_evidence/'+str(index))
        index11=0
        for head1,link1,tag1 in zip(head,link,img_tag1):
            try:
                img_text=get_text_only(link1)
                context=find_context(img_text,keywords11)
                print('图片相关上下文:',context)
                path='./img_evidence/'+str(index)+'/'+str(index11)+'.jpg'
                save_base64_image(tag1, path)
        #         if context!='':
                data1=[str(index11),head1,link1,context]
                img_evi.append(f"({'，'.join(data1)})")
                index11=index11+1
            except:
                index11=index11+1
                continue
        df.at[index, 'img_evidence'] = f"({'。'.join(img_evi)})"
        print(f"({'。'.join(img_evi)})")
    except:
        pass
#     k=0
#     path_list=[]
    
#     if not os.path.exists('./img_evidence/'+str(index)):
#         os.makedirs('./img_evidence/'+str(index))
#     for link1 in img_tag1:
#         path='./img_evidence/'+str(index)+'/'+str(k)+'.jpg'
#         save_base64_image(link1, path)
#         path_list.append(path)
#         k=k+1
    start_time3 = time.time()
    print('用时：',start_time3-start_time2)
    if index%20==0:
        df.to_csv('snope1512_'+str(index)+'.csv',index=False)