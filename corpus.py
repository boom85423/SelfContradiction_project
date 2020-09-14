import json
import pandas as pd
import pyprind
import mwparserfromhell
import numpy as np
from nltk.tokenize import sent_tokenize
import re
import pickle
import sys


# with open('./dataset/solved.jsonl', 'r') as jsonlfile:
#     jsonlfile = list(jsonlfile)
# 
# title, X, Y = [], [], []
# for i in pyprind.prog_bar(range(len(jsonlfile))):
#     a = json.loads(jsonlfile[i])
#     text = a['povVersion']
#     solve_text = a['solvedpovVersion']
#     pageTitle = a['pageTitle']
#     
#     wikicode = mwparserfromhell.parse(text)
#     templates = wikicode.filter_templates()
#     label_pov = 0
#     for j in range(len(templates)):
#         if 'Self-contradictory' in templates[j]: 
#             label_pov = 1
# 
#     if len(text.split()) != 0:
#         title.append(pageTitle)
#         X.append(str(text))
#         Y.append(label_pov)
# 
#     wikicode = mwparserfromhell.parse(solve_text)
#     templates = wikicode.filter_templates()
#     label_solved_pov = 0
# 
#     if len(solve_text.split()) != 0:
#         title.append(pageTitle)
#         X.append(str(solve_text))
#         Y.append(label_solved_pov)


# title, revision_id, X, Y = [], [], [], []
# selfC = pd.read_csv('./dataset/selfC.csv') # (5531,4):fixed contradictory part and revised other parts in page
# solvedSelfC = pd.read_csv('./dataset/solvedSelfC.csv') # (280,5):non-contradiction with id of the last edited history
# 
# for i in pyprind.prog_bar(range(len(selfC))):
#     title_i = list(selfC['page_title'])[i]
#     r_id = list(selfC['revision_id'])[i]
#     text = list(selfC['revision_text'])[i]
#     if isinstance(text,str) is True and len(text.split()) !=0:
#         wikicode = mwparserfromhell.parse(text)
#         templates = wikicode.filter_templates()
#         label = 0
#         for j in range(len(templates)):
#             if 'Self-contradictory' in templates[j]:
#                 label = 1
#     title.append(title_i)
#     revision_id.append(r_id)
#     X.append(str(text))
#     Y.append(label)
# 
# for i in pyprind.prog_bar(range(len(solvedSelfC))):
#     title_i = list(solvedSelfC['page_title'])[i]
#     r_id = list(selfC['revision_id'])[i]
#     text = list(solvedSelfC['revision_text'])[i]
#     if isinstance(text,str) is True and len(text.split()) !=0:
#         wikicode = mwparserfromhell.parse(text)
#         templates = wikicode.filter_templates()
#         label = 0
#     title.append(title_i)
#     revision_id.append(r_id)
#     X.append(str(text))  
#     Y.append(label)
# 
# file = open('./dataset/corpus_before', 'wb')
# pickle.dump({"title":title, "revision_id":revision_id, "X":X, "Y":Y}, file)
# file.close()
# sys.exit(0)

with open('./dataset/corpus_before', 'rb') as file:
    corpus = pickle.load(file)
title = corpus['title']
revision_id = corpus['revision_id']
X = corpus['X']
Y = corpus['Y']

## Clean the corpus
X_clear = []
for i in pyprind.prog_bar(range(len(X))):
# for i in pyprind.prog_bar([4096]):
    text = X[i]
    
    wikicode = mwparserfromhell.parse(text)
    tags = wikicode.filter_tags()

    ## Remove external links
    wikicode = mwparserfromhell.parse(text)
    external_links = wikicode.filter_external_links()
    str_wikicode = str(wikicode)
    for j in range(len(external_links)):
        str_wikicode = str_wikicode.replace(str(external_links[j]),'')

    ## Remove comments
    wikicode = mwparserfromhell.parse(text)
    comments = wikicode.filter_comments()
    str_wikicode = str(wikicode)
    for j in range(len(comments)):
        str_wikicode = str_wikicode.replace(str(comments[j]),'')

    ## Remove templates
    wikicode = mwparserfromhell.parse(str_wikicode)
    templates = wikicode.filter_templates()
    infobox = []
    for j in range(len(templates)):
        if 'Infobox' in templates[j]:
            ## Should be paste to article latter
            infobox_tmp = mwparserfromhell.parse(templates[j]).filter_text()
            infobox_tmp = [str(k) for k in infobox if len(k.split()) > 0 and "http" not in k]
            infobox_tmp = " ".join(infobox).split('\n')
            infobox = infobox_tmp
            str_wikicode = str_wikicode.replace(str(templates[j]),'')
        elif '{{convert' in templates[j]:
            convert = mwparserfromhell.parse(templates[j]).filter_text()
            convert = [str(k) for k in convert[1:]]
            convert = " ".join(convert)
            str_wikicode = str_wikicode.replace(str(templates[j]), convert)
        else:
            str_wikicode = str_wikicode.replace(str(templates[j]),'')
   
    ## Keep blod type words instead of filter those tags
    str_wikicode = str_wikicode.replace("'''",'')
    str_wikicode = str_wikicode.replace("''",'')

    ## Grasp the last wikilink
    wikicode = mwparserfromhell.parse(str_wikicode)
    wikilinks = wikicode.filter_wikilinks()
    str_wikicode = str(wikicode)
    for j in range(len(wikilinks)):
        entity = str(wikilinks[j]).split('|')[-1]
        if re.search('.px', entity) == True:
            entity = ''
        str_wikicode = str_wikicode.replace(str(wikilinks[j]), entity)
    
    ## Remove headings
    wikicode = mwparserfromhell.parse(str_wikicode)
    headings = wikicode.filter_headings()
    str_wikicode = str(wikicode)
    for j in range(len(headings)):
        if "See also" in headings[j]:
            str_wikicode = str_wikicode.split(str(headings[j]))[0]
        elif "References" in headings[j]:
            str_wikicode = str_wikicode.split(str(headings[j]))[0]
        elif "External links" in headings[j]:
            str_wikicode = str_wikicode.split(str(headings[j]))[0]
        else:
            str_wikicode = str_wikicode.replace(str(headings[j]),'')
    
    ## Deal with some weired dot-data
    str_wikicode = str_wikicode.replace('******', '*')
    str_wikicode = str_wikicode.replace('*****', '*')
    str_wikicode = str_wikicode.replace('****', '*')
    str_wikicode = str_wikicode.replace('***', '*')
    str_wikicode = str_wikicode.replace('**', '*')
    
    ## Remove tags
    wikicode = mwparserfromhell.parse(str_wikicode)
    tags = wikicode.filter_tags()
    for j in range(len(tags)):
        if ('*' in tags[j]) or ('#' in tags[j]) or ('| ' in tags[j]) or ('|' in tags[j]) or ('||' in tags[j]):
            pass
        elif ('-' in tags[j]):
            if ('ref' in tags[j]):
                str_wikicode = str_wikicode.replace(str(tags[j]),'')
            else:
                pass
        else:
            str_wikicode = str_wikicode.replace(str(tags[j]),'')
    ## In some case, you may need filter again to get the clear wikidata
    wikicode = mwparserfromhell.parse(str_wikicode)
    tags = wikicode.filter_tags()
    for j in range(len(tags)):
        if ('*' in tags[j]) or ('#' in tags[j]) or ('| ' in tags[j]) or ('|' in tags[j]) or ('||' in tags[j]):
            pass
        elif ('-' in tags[j]): 
            if ('ref' in tags[j]):
                str_wikicode = str_wikicode.replace(str(tags[j]),'')
            else:
                pass
        else:
            str_wikicode = str_wikicode.replace(str(tags[j]),'')
    wikicode = mwparserfromhell.parse(str_wikicode)
    tags = wikicode.filter_tags()

    ## Keep the table intact by droping the style before using sent_tokenize
    str_wikicode = str_wikicode.replace('style="background', '')
    str_wikicode = re.sub('bgcolor="\#.*?"', ' ', str_wikicode)
    str_wikicode = re.sub('\#.*?;', ' ', str_wikicode)
    str_wikicode = str_wikicode.replace('! #', '')
    str_wikicode = str_wikicode.replace('!#', '')
   
    ## Paste infobox
    str_wikicode_sent = sent_tokenize(str_wikicode)
    str_wikicode_sent = infobox + str_wikicode_sent
    
    ## Split dot data as multiple sentences
    dot, dot_idx = [], []
    str_wikicode_sent = np.array(str_wikicode_sent)
    for j in str_wikicode_sent:
        if ('*' in j) or ('#' in j) or (' – ' in j):
            idx = np.where(str_wikicode_sent == j)[0]
            dot.append(j)
            dot_idx += idx.tolist()
    if len(dot_idx) > 0:
        str_wikicode_sent = np.delete(str_wikicode_sent, np.array(dot_idx))
        dot_sentences = []
        for j in dot:
            dot_sentences += j.split('\n')
        str_wikicode_sent = str_wikicode_sent.tolist()
        str_wikicode_sent += dot_sentences
   
    ## Split table data as multiple sentences
    table, table_idx = [], []
    str_wikicode_sent = np.array(str_wikicode_sent)
    for j in str_wikicode_sent:
        if ('|-' in j) or ('|}' in j):
            idx = np.where(str_wikicode_sent == j)[0]
            table.append(j)
            table_idx += idx.tolist()
    
    def remove_style(string):
        if 'style="text-align:center;"' in string:
            string = string.replace('style="text-align:center;"', '')
        elif 'style="text-align:left;"' in string:
            string = string.replace('style="text-align:left;"', '')
        elif 'style="text-align:right;"' in string:
            string = string.replace('style="text-align:right;"', '')
        elif 'style="text-align: center;"' in string:
            string = string.replace('style="text-align: center;"', '')
        elif 'style="text-align: left;"' in string:
            string = string.replace('style="text-align: left;"', '')
        elif 'style="text-align: right;"' in string:
            string = string.replace('style="text-align: right;"', '')
        elif 'align="right"' in string:
            string = string.replace('align="right"', '')
        elif 'align="left"' in string:
            string = string.replace('align="left"', '')
        elif 'align="center"' in string:
            string = string.replace('align="center"', '')
        elif 'style="background:' in string:
            string = string.replace('style="background:', '')
        return string
    
    if len(table_idx) > 0:
        str_wikicode_sent = np.delete(str_wikicode_sent, np.array(table_idx))
        table_sentences = []
        for j in table:
            if '|-' in j:
                for k in j.split('|-'):
                    k = remove_style(k)
                    table_sentences.append(k)
            elif '|}' in j:
                for k in j.split('|}'):
                    k = remove_style(k)
                    table_sentences.append(k)
        
        str_wikicode_sent = str_wikicode_sent.tolist()
        str_wikicode_sent += table_sentences

    article = []
    for j in range(len(str_wikicode_sent)):
        sent = str_wikicode_sent[j]
        sent = ' '.join(re.sub('width=".*?"',' ',sent).split())
        sent = ' '.join(re.sub('height=".*?"',' ',sent).split())
        sent = ' '.join(re.sub('[^a-zA-Z0-9.,]',' ',sent).split())
        if len(sent.split()) > 1:
            article.append(sent)
    X_clear.append(article)

## Check num_words before splitting
# num_words = []
# page_idx = []
# for i in pyprind.prog_bar(range(len(X_clear))):
#     num_words += [len(X_clear[i][j].split()) for j in range(len(X_clear[i]))]
#     for j in range(len(X_clear[i])):
#         if len(X_clear[i][j].split()) > 200:
#             page_idx.append(i)
# num_words = np.array(num_words)
# max_words = np.max(num_words)
# print(np.min(num_words), np.median(num_words), np.max(num_words))
# print(set(page_idx))

## Split train and test by title
train_titles = np.random.choice(list(set(title)), int(len(set(title))*0.8), replace=False)
test_titles = np.array(list(set(title) - set(train_titles)))
train_idx = []
for i in train_titles:
    train_idx += np.where(np.array(title) == i)[0].tolist()
test_idx = []
for i in test_titles:  
        test_idx += np.where(np.array(title) == i)[0].tolist()

## Sample a balanced data
train_X_title = [title[i] for i in train_idx]
test_X_title = [title[i] for i in test_idx]
train_X_raw = [X_clear[i] for i in train_idx]
test_X_raw = [X_clear[i] for i in test_idx]
train_Y = [Y[i] for i in train_idx]
test_Y = [Y[i] for i in test_idx]
train_revision_id = [revision_id[i] for i in train_idx]
test_revision_id = [revision_id[i] for i in test_idx]

train_Y_neg_idx = np.where(np.array(train_Y) == 0)[0]
train_Y_pos_idx = np.where(np.array(train_Y) == 1)[0]
num_balance_train = min(len(train_Y_neg_idx), len(train_Y_pos_idx))
test_Y_neg_idx = np.where(np.array(test_Y) == 0)[0]
test_Y_pos_idx = np.where(np.array(test_Y) == 1)[0]
num_balance_test = min(len(test_Y_neg_idx), len(test_Y_pos_idx))
train_idx_balanced = np.concatenate((np.random.choice(train_Y_neg_idx, num_balance_train, replace=False), np.random.choice(train_Y_pos_idx, num_balance_train, replace=False))) 
test_idx_balanced = np.concatenate((np.random.choice(test_Y_neg_idx, num_balance_test, replace=False), np.random.choice(test_Y_pos_idx, num_balance_test, replace=False)))

train_X_title = [train_X_title[i] for i in train_idx_balanced]
train_X_raw = [train_X_raw[i] for i in train_idx_balanced] 
train_Y = [train_Y[i] for i in train_idx_balanced]
test_X_title = [test_X_title[i] for i in test_idx_balanced]
test_X_raw = [test_X_raw[i] for i in test_idx_balanced]
test_Y = [test_Y[i] for i in test_idx_balanced]
train_revision_id = [train_revision_id[i] for i in train_idx_balanced]
test_revision_id = [test_revision_id[i] for i in test_idx_balanced]

## filter pages that have unsufficent sentences for building SCC
train_idx_SCC = np.array([len(i) for i in train_X_raw])
train_idx_SCC = np.where(train_idx_SCC >= 3)[0]
test_idx_SCC = np.array([len(i) for i in test_X_raw])
test_idx_SCC = np.where(test_idx_SCC >= 3)[0]

train_X_title = [train_X_title[i] for i in train_idx_SCC]
train_X_raw = [train_X_raw[i] for i in train_idx_SCC]
train_Y = [train_Y[i] for i in train_idx_SCC]
test_X_title = [test_X_title[i] for i in test_idx_SCC]
test_X_raw = [test_X_raw[i] for i in test_idx_SCC]
test_Y = [test_Y[i] for i in test_idx_SCC]
train_revision_id = [train_revision_id[i] for i in train_idx_SCC]
test_revision_id = [test_revision_id[i] for i in test_idx_SCC]

file = open('./dataset/corpus_clear_splitedByTitleBalanced.pickle', 'wb')
pickle.dump({"train_X_title":train_X_title, "train_X_raw":train_X_raw, "train_Y":train_Y, "train_revision_id":train_revision_id,\
             "test_X_title":test_X_title, "test_X_raw":test_X_raw, "test_Y":test_Y, "test_revision_id":test_revision_id}, file)
file.close()
