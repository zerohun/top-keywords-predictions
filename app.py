import pyspark
import json
import re
from langdetect import detect

from textblob import TextBlob
import nltk
from functools import reduce

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def expForWordsFilter(targetStr, filterList):
    return re.compile('|'.join(map(lambda word: word, filterList)))

def removeWordsByList(targetStr, filterList):
    exp = expForWordsFilter(targetStr, filterList)
    print(exp)
    return exp.sub('', targetStr)

def combineTable(a, b):
    result = {}
    for k in a:
        result[k] = a[k]
    for k in b:
        if k in result:
            result[k] += b[k]
        else:
            result[k] = b[k]

    return result

WHITELIST_CHARS = [
    '[A-z]',
    '\''
]

WORDS_TO_FILTER_OUT = [
    '[@]',
    '[RT]',
    '[&amp;]',
    '(^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$)',
    '(\".+\")',
    '(\'.+\')'

]
sc = pyspark.SparkContext.getOrCreate()
exp = re.compile('[a-z]+')

jsonFile = sc.textFile("./resources/29.json")
jsonObjs = jsonFile.flatMap(lambda wholeText:wholeText.split('\n')).map(lambda line:json.loads(line)).filter(lambda text:'text' in text)
texts = jsonObjs.map(lambda jsonObj:jsonObj["text"])
filteredTexts = texts.map(lambda text:"".join(re.compile('[A-z]|[\']|[\s]').findall(text))).map(lambda text:removeWordsByList(text, WORDS_TO_FILTER_OUT)).filter(lambda text: len(reduce(lambda a,b:a+b, exp.findall(text), '')) > (len(text)) / 1.5).filter(lambda text: detect(text) == 'en')
taggedWords = filteredTexts.flatMap(lambda text:TextBlob(text).tags)
keywords = taggedWords.filter(lambda word:('NN' in word[1]) and len(word[0]) > 1).map(lambda word:word[0].lower())
countTable = keywords.map(lambda keyword:{keyword: 1}).reduce(lambda a,b:combineTable(a,b))
import operator
sortedList = sorted(countTable.items(), key=operator.itemgetter(1))
sortedList.reverse()
print(sortedList)





