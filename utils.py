from sklearn.metrics import confusion_matrix
from nltk.tokenize import sent_tokenize
import mwparserfromhell
import re
import pyprind
import numpy as np
from sklearn.metrics import f1_score


def Evaluate(Y, Y_hat):
    TP,FP,FN,TN = 0,0,0,0
    for i in range(len(Y_hat)):
        if int(Y_hat[i]) == 1 and int(Y[i]) ==1:
            TP+=1
        elif int(Y_hat[i]) == 1 and int(Y[i]) ==0:
            FP +=1
        elif int(Y_hat[i]) == 0 and int(Y[i]) ==1:
            FN +=1
        elif int(Y_hat[i]) == 0 and int(Y[i]) ==0:
            TN +=1
        else:
            print('[ERROR]')
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Precision = (TP)/(TP+FP)
    Recall = (TP)/(TP+FN)
    F1 = f1_score(Y, Y_hat)

    print('acc: %.4f, precision: %.4f, recall: %.4f, f1: %.4f' % (Accuracy, Precision, Recall, F1))
    print(confusion_matrix(Y, Y_hat))


def clear_sentences(X):
    X_clear = []
    available_idx = []
    for i in pyprind.prog_bar(range(len(X))):
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

        if len(str_wikicode_sent) > 1:
            concat = ""
            for i in range(len(str_wikicode_sent)):
                concat += str_wikicode_sent[i]
            str_wikicode_sent = [concat]

        article = []
        for j in range(len(str_wikicode_sent)):
            sent = str_wikicode_sent[j]
            sent = ' '.join(re.sub('width=".*?"',' ',sent).split())
            sent = ' '.join(re.sub('height=".*?"',' ',sent).split())
            sent = ' '.join(re.sub('[^a-zA-Z0-9.,]',' ',sent).split())
            if len(sent.split()) > 1:
                article.append(sent)
                available_idx.append(i)
        X_clear.extend(article)
    return X_clear, available_idx

