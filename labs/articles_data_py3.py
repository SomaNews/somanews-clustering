# -*- coding: utf-8 -*- 

import numpy as np
import pandas as pd
import datetime
from gensim.models import Word2Vec
import cnouns as cn
import pickle
from multiprocessing import Pool
from time import time
import re
import os


def remove_headlines(text, headline_path):
    headlines = pd.read_pickle(headline_path)
    headlines = headlines['headline'].tolist()
    result = re.match(r"[^[]*\[([^]]*)\]", text)
    
    if result:
        if result.groups()[0] == '경향포토':
            text = text.replace(text, 'NaN')
            return text
        
        for headline in headlines:
            text = text.replace(headline, '')
        
    return text

def get_target_cate():
    return [u"정치", u"사회", u"과학", u"경제"]

def find_recent_articles(collection, catelist_path):
    articles = collection

    categories = pd.read_pickle(catelist_path)

    article_list = []
    d = datetime.datetime.now() - datetime.timedelta(days=7)
    for article in articles.find({"publishedAt": {"$gt": d}}).sort("publishedAt"):
        article_list.append(article)

    articles_df = pd.DataFrame.from_dict(article_list)

    new_categories = []

    for idx, row in articles_df.iterrows():
        category = categories[categories.category==row.category]
        if(len(category) > 0):
            new_categories.append(category['name'].iloc[0])
        else:
            new_categories.append('none')

    articles_df['cate'] = new_categories
    target_list = get_target_cate()
    
    return articles_df[articles_df['cate'].isin(target_list)]

class Sentences(object):
    def __init__(self, dirname, size):
        self.dirname = dirname
        self.size = size
 
    def __iter__(self):
        for fname in os.listdir(self.dirname)[:self.size]:
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
                                
def makeDataset(collection, target_dir, corpus_path, batch_size=5000, workers=4, tokenize=cn.pos_tags):
    articles = collection.find()
    
    articles_list = []
    for article in articles:
        articles_list.append(article)
    articles_df = pd.DataFrame.from_dict(articles_list)
    print("Number of articles - %d" % len(articles_df))
    
    corpus_df = pd.read_pickle(corpus_path)
    print("Number of corpus - %d" % len(corpus_df))
    
    corpus_words = [row[1] for row in corpus_df.iteritems()]
    articles_words = [aricle['title'] + ' ' + aricle['content'] for idx, aricle in articles_df.iterrows()]
    words = corpus_words + articles_words
    corpus_words = []
    articles_words = []
    print("Number of words - %d" % len(words))
    
    batchs = [words[i:i + batch_size] for i in xrange(0, len(words), batch_size)]
    print("Number of batchs - %d" % len(batchs))
    
    # p = Pool(1)
    for idx, batch in enumerate(batchs):
        t0 = time()
        # tokens = p.map(tokenize, batch)
        tokens = [tokenize(b) for b in batch]
        f = open("%s/%d"%(target_dir, idx), "w")
        f.write("\n".join(tokens).encode('utf8'))
        f.close()
        print("Batch#%d - tokenize took %f sec"%(idx, time() - t0))
        
    return len(batchs)
        
def trainWord2Vec(src, dest, size=50):
    sentences = Sentences(src, size)
    w2v = Word2Vec(sentences)
    w2v.save_word2vec_format(dest)