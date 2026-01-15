import pandas as pd
from sklearn.model_selection import train_test_split
all_df=pd.read_csv('snope1512_1512_index_no2.csv')
train_df, val_df = train_test_split(all_df, test_size=0.3, random_state=42,stratify=all_df['fine-grained label'])
import re
import numpy as np
web1=[]
for index, row in all_df.iterrows():
    text=row['text_evidence']
#     print(text)
    if pd.notna(text):
        domains = re.findall(r"https?://([^/\\]+)", text)
        web1 = list(set(web1) | set(domains))
for index, row in all_df.iterrows():
    text=row['img_evidence']
#     print(text)
    if pd.notna(text):
        domains = re.findall(r"https?://([^/\\]+)", text)
        web1 = list(set(web1) | set(domains))
print(len(web1))
# Define category sets
fact_checking = {
    'fullfact.org','www.verifythis.com','pesacheck.org','factcheck.org','politifact.com',
    'snopes.com','mythdetector.com','truthorfiction.com','namibiafactcheck.org.na',
    'srilanka.factcrescendo.com','snopes.com','factcheck.org','politifact.com','pesacheck.org','mythdetector.com',
    'namibiafactcheck.org.na','srilanka.factcrescendo.com','truthorfiction.com','sn.eferrit.com','factuel.afp.com','www.verifythis.com','factcheck.org',
    'politifact.com','pesacheck.org','mythdetector.com','namibiafactcheck.org.na', 'sn.eferrit.com','factuel.afp.com','www.verifythis.com','factcheck.org',
    'politifact.com','pesacheck.org','mythdetector.com','srilanka.factcrescendo.com'
}

# 社区论坛站点（示例，按需补充）
community = {
    'archive.thetab.com','www.denisonforum.org','www.dailykos.com','theboar.org',
    'mikemcguff.blogspot.com','thoughtcatalog.com','mouseplanet.com','knowyourmeme.com','archive.thetab.com','www.denisonforum.org','www.dailykos.com','theboar.org',
    'thoughtcatalog.com','m.imdb.com','mouseplanet.com','knowyourmeme.com','tr.pinterest.com','vk.com','m.facebook.com','imgflip.com','watch.plex.tv',
    'hero.fandom.com','steemit.com','mysingingmonsters.fandom.com','tr.pinterest.com','vk.com','m.facebook.com','imgflip.com','watch.plex.tv',
    'hero.fandom.com','steemit.com','mysingingmonsters.fandom.com','memeguy.com',
    'paper.stheadline.com','community.naimaudio.com','community.robotshop.com'
}

# 视频分享站点（示例，按需补充）
video_sharing = {
    'youtube.com','vimeo.com','m.imdb.com','www.livenowfox.com','youtu.be','vimeo.com','dailymotion.com','open.spotify.com','youtu.be','m.youtube.com','open.spotify.com','dailymotion.com','youtu.be','m.youtube.com','open.spotify.com','dailymotion.com','tubitv.com','tk.com'
}

# 其他类别（示例，按需补充）
other_categories = {
    'www.bmw.com','www.adidas-group.com','www.amazon.com','www.ebay.com','www.bmw.com','www.adidas-group.com','archive.starbucks.com','www.amazon.co.uk','www.ebay.com','www.bmw.com','www.adidas-group.com',    'archive.starbucks.com','www.amazon.co.jp','www.ebay.com','www.bmw.com','www.adidas-group.com',
    'photos.com','issuu.com','books.google.com','pikbest.com'
}

mapping = {
    d: (
        # 0.9       if d in fact_checking else
        0       if d in fact_checking else
        0.2    if d in community     else
        0.3  if d in video_sharing else
        0.5     if d in other_categories else
        0.8
    )
    for d in web1
}
def verify_text(text):
    text_model=model1
    text_model.load_state_dict(torch.load(r'E:\finefake\FineFake\model1.bin'))
    verify1=text_model(text)
    return verify1
def verify_img(img):
    img_model=model2
    img_model.load_state_dict(torch.load(r'E:\finefake\FineFake\model2.bin'))
    verify2=img_model(img)
    return verify2
def filter_img_evidence(parts):
    filter_index,filter_text='',''
    if parts!='()' and pd.notna(parts):
        parts=re.split(r'\)\。\(', parts[1:-1]) 
        max_score=0
	max_score2=0
        # parts = train_df['img_evidence'][965][1:-1].split('。')
#         print(parts)
        for part in parts:
    #         s_part=part[1:-1].split('，',3)
            try:
                before_url, url, context = part[1:-1].rsplit("，", 2)
                index, title = before_url.split("，", 1)
                k=re.findall(r"https?:\/\/([^\/]+)", part)
                if k!=[] and index!='':
                    score=mapping[k[0]]
                    if score>=max_score:
                        max_score=score
			img=Image.open('./img_evidence/'+str(index)+'/'+filter_img+'.jpg')
			score2=verify_img(img)
			if score2>max_score2:
				max_score2=score2
                        	filter_index,filter_text=index,context
            except:pass
        return filter_index,filter_text
    else:return filter_index,filter_text
def filter_text_evidence(parts):
    filter_index,filter_text='',''
    if parts!='()' and pd.notna(parts):
#         print(parts)
        parts=re.split(r'\)\。\(', parts[1:-1]) 
        max_score=0
        # parts = train_df['img_evidence'][965][1:-1].split('。')
#         print(parts)
        for part in parts:
    #         s_part=part[1:-1].split('，',3)
            try:
                index, url,tf, after = part[1:-1].split("，", 3)
                context, time_bet = after.rsplit("，", 1)
                k=re.findall(r"https?:\/\/([^\/]+)", part)
                if k!=[] and context!="":
                    score=mapping[k[0]]
                    if score>max_score:
                        max_score=score
			score2=verify_text(context)
			if score2>max_score2:
				max_score2=score2
                        	filter_index,filter_text=index,context
            except:pass
        if max_score==0:
            filter_index,filter_text=None,None
        return filter_index,filter_text
    return filter_index,filter_text
a,b=filter_text_evidence(all_df['text_evidence'][0])
print(a)
print(b)
import re
rate_l=set()
for index, row in all_df.iterrows():
    text=row['knowledge_search']
    if pd.notna(text):
        inner = text[1:-1]

        # 2. 在 “) ， (” 处切开，得到两部分
        parts = re.split(r"\)\s*，\s*\(", inner)

        # 3. 清理每段开头结尾的括号（如果有的话）
        parts = [p.strip("()") for p in parts]
        if parts!=['']:
#             print(parts)
            for part in parts:
#                 print(part)
                text,rate,url=part.rsplit('，',2)
#                 print(rate)
                rate_l.add(rate)
# print(rate_l)
def check_string(s):

    lower_s = s.lower()

    return 'true' in lower_s or 'correct' in lower_s or '正確' in lower_s
def knowledge_search(text):
    if pd.notna(text):
        inner = text[1:-1]

        parts = re.split(r"\)\s*，\s*\(", inner)
        parts = [p.strip("()") for p in parts]
        if parts!=['']:
            for part in parts:
                text,rate,url=part.rsplit('，',2)
                if check_string(rate):
                    return 0
                else:return 1
        else:return 2
# 0为正确1为错误2为未搜到
    else:return 2
# check_string('true1')
def verify_text(text):
    text_model=model1
    text_model.load_state_dict(torch.load(r'E:\finefake\FineFake\model1.bin'))
    verify1=text_model(text)
    return verify1
def verify_img(img):
    img_model=model2
    img_model.load_state_dict(torch.load(r'E:\finefake\FineFake\model2.bin'))
    verify2=img_model(img)
    return verify2

knowledge_search(all_df['knowledge_search'][0])
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
class NewsDataset(Dataset):
    def __init__(self, data,num_classes=4):
        self.df=data
        self.image_folder ='./'
        self.num_classes = num_classes
#         _,self.preprocess = clip.load("ViT-B/32")
        self.tokenizer=AutoTokenizer.from_pretrained(r'./model/bert-base-uncased')
        self.transform= transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图片大小
        transforms.ToTensor(),  # 将图片转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        text,image_path,label,knowledge,index = row['text'],row['image_path'],row['fine-grained label'],row['knowledge_search'],row['index']
        _,filter_text=filter_text_evidence(row['text_evidence'])
        filter_img,filter_img_text=filter_img_evidence(row['img_evidence'])
        knol_result=knowledge_search(row['knowledge_search'])
        # print(row['img_evidence'])
        # print('@')
        # print(filter_img)
        image_name=row['image_path']
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        # 应用转换（如调整大小、归一化等）
        image = self.transform(image)
        if filter_img!='':
            filter_img=Image.open('./img_evidence/'+str(index)+'/'+filter_img+'.jpg')
            filter_img = self.transform(filter_img)
        else:filter_img=image
#         image_path = self.preprocess(Image.open(image_path)).unsqueeze(0)
        label=row['fine-grained label']
        text=self.tokenizer(text, max_length=128, padding='max_length', truncation=True)
        filter_text=self.tokenizer(filter_text, max_length=400, padding='max_length', truncation=True)
        filter_img_text=self.tokenizer(filter_img_text, max_length=400, padding='max_length', truncation=True)
        
        text = {k: torch.tensor(v) for k, v in text.items()}
        filter_text = {k: torch.tensor(v) for k, v in filter_text.items()}
        filter_img_text = {k: torch.tensor(v) for k, v in filter_img_text.items()}
        image_name=row['image_path']
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        # 应用转换（如调整大小、归一化等）
        image = self.transform(image)
        if label==4:
            label=2
        # print({
        #     'text':text,
        #     'image': image,
        #     'text_evidence':filter_text,
        #     'image_evidence_img': filter_img,
        #     'image_evidence_text': filter_img_text,
        #     'kb_signal':knol_result,
        #     # 多任务标签
        #     'text_label': 1 if label == 3 else 0,
        #     'image_label': 1 if label == 2 else 0,    # 图像真伪标签 0/1
        #     'align_label': 1 if label == 1 else 0,    # 图文对齐标签 1(一致)/0(不一致)
        #     'final_label': label,    # 最终四分类标签
        #     'index':index
        # })
        # print(knol_result)
        return {
            'text':text,
            'image': image,
            'text_evidence':filter_text,
            'image_evidence_img': filter_img,
            'image_evidence_text': filter_img_text,
            'kb_signal':torch.tensor(knol_result),
            # 多任务标签
            'text_label': torch.tensor(1 if label == 3 else 0),
            'image_label': torch.tensor(1 if label == 2 else 0),    # 图像真伪标签 0/1
            'align_label': torch.tensor(1 if label == 1 else 0),    # 图文对齐标签 1(一致)/0(不一致)
            'final_label': torch.tensor(label)    # 最终四分类标签
            # 'index':index
        }
train_dataset=NewsDataset(train_df)
val_dataset=NewsDataset(val_df)
# a[0]
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
#     collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=True,
#     collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)
