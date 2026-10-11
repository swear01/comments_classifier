# https://github.com/Cheng-Yi-Ting/LDA/blob/master/lda.py
import pysnooper
import jieba
import jieba.posseg as pseg
import json
import re
import pickle
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from lda_storage import load_model

path = './politics/big/'
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
    with open(f'word_matrix.pickle','rb') as f :
        data = pickle.load(f)

    lda = load_model('lda_model20.pickle')
    
    corpus = data.seg_docs
    # 將文本中的詞語，轉換成詞頻矩陣
    vectorizer = CountVectorizer()
    # print(vectorizer)
    # 計算詞語出現的頻率
    tf = vectorizer.fit_transform(corpus)
    vocabrary = vectorizer.vocabulary
    # print("###################################")
    # 獲取詞袋中所有文本關鍵字，詞袋中所有的字詞
    words = vectorizer.get_feature_names()

    with open("0.json","r",encoding="UTF-8") as f :
        submission = json.load(f)['comments']
    comments = [" ".join(jieba.lcut(i['comment'])) for i in submission]
    raw_comments = [i['comment'] for i in submission]
    #print(comments)
    term_freq = vectorizer.transform(comments)
    chance_matrix = lda.transform(term_freq)
    print(type(chance_matrix))
    max_ids = list(zip(np.argmax(chance_matrix,axis=0),np.amax(chance_matrix,axis=0))) 
    print(max_ids)
    id_set = set()
    for max_id in max_ids :
        if max_id[1] < 0.1 : continue
        id_set.add(max_id[0])

    abstract = [raw_comments[i] for i in id_set]
    print(abstract)

    bucket = [list() for i in range(len(chance_matrix[0]))]
    for comment,chance in zip(raw_comments,chance_matrix) :
        if np.amax(chance) < 0.5 : continue
        bucket[np.argmax(chance)].append(comment) 
    print(bucket[2])

    with open(f"./LDAcomments.json","w",encoding="utf-8") as f :
        json.dump(bucket,f,indent=2,ensure_ascii=False)  
        

    
