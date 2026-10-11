# https://github.com/Cheng-Yi-Ting/LDA/blob/master/lda.py
import pysnooper
import jieba
import jieba.posseg as pseg
import json
import re
import pickle
from collections import OrderedDict
from tqdm import tqdm

from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from lda_storage import save_model

path = './politics/big/'
subjects = 10
data_pickle = 'word_matrix.pickle'
'''
def data_in():
    data = []
    for index in range(0,36) :
        with open(f"{path}{index}00.json","r",encoding="utf-8") as f :
            subs= json.load(f) 
        for sub in subs :
            combined = "".join([comment['comment'] for comment in sub['comments']]) 
            data.append(combined)
    return data
'''
#@pysnooper.snoop()
class Data():
    def __init__(self):
        self.docs = {}
        with open(f"disable_pharse.txt","r",encoding="utf-8") as f :
            disable_phase= f.read().split("\n")
        self.stopword = disable_phase
        #print(self.stopword)
        #raise UserWarning
        self.seg_docs = self.get_seg_docs()

    def get_seg_docs(self):
        _seg_docs = []
        for file_num in tqdm(range(0,36)) :
            with open(f"{path}{file_num}00.json","r",encoding="utf-8") as f :
                sub_docs = json.load(f) 
            for sub in sub_docs :
                content_str = ''
                words = " ".join([comment['comment'] for comment in sub['comments']])
                for w in jieba.lcut(words):
                    if w not in self.stopword and re.search(u'[\u4e00-\u9fff]', w) :
                        content_str = content_str+' '+w
                        # content_seg.append(w)
                #print(content_str)
                _seg_docs.append(content_str)
        return _seg_docs


if __name__ == '__main__':
    jieba.load_userdict("zh_tw_dict.txt")
    with open(data_pickle,"rb",) as f :
        data = pickle.load(f)
    # print(data.seg_docs)
    # corpus = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
    #           "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
    #           "小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
    #           "我 爱 北京 天安门"]  # 第四类文本的切词结果
    corpus = data.seg_docs
    # 將文本中的詞語，轉換成詞頻矩陣
    vectorizer = CountVectorizer()
    # print(vectorizer)
    # 計算詞語出現的頻率
    tf = vectorizer.fit_transform(corpus)
    # print("###################################")
    # 獲取詞袋中所有文本關鍵字，詞袋中所有的字詞
    words = vectorizer.get_feature_names()


    lda = LatentDirichletAllocation(
        n_components=subjects, learning_offset=50., random_state=0)
    docres = lda.fit_transform(tf)
    # print("###################################")
    # 文檔-主題分佈矩陣
    print(docres)
    # print("###################################")
    # 主題-詞語分佈矩陣
    print(lda.components_)
    # 印出每個主題下權重教中的字詞
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-subjects - 1:-1]]))

    save_model(lda, f'lda_model{subjects}.pickle')
