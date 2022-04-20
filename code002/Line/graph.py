import re
import os
from nltk.tokenize import word_tokenize

from string import punctuation



def getAllAttributes(data, name):
    thispath = 'data/'
    if not os.path.exists(thispath):
        os.mkdir(thispath)
    namepath = thispath+name+"/"
    if not os.path.exists(namepath):
        os.mkdir(namepath)

    def getAbstract(namepath, data, name):

        originalData = data.split('\n\n')
        outputfileGraph = open(namepath +'abstractGraph.txt', 'w')
        outputfileCoauthors = open(namepath + 'abstractList.txt', 'w')
        abstractList = []
        for i in range(0, len(originalData)):
            if originalData[i] != '':
                abstract = re.search('abstract:[^\n]*', originalData[i]).group()[9:].split('|')
                for k in range(len(abstract)):
                    abstract[k] = abstract[k].lower()
                    if abstract[k] in abstractList:
                        outputfileGraph.write(str(i) + ' ' + str(abstractList.index(abstract[k]) + 10000) + '\n')
                    else:
                        abstractList.append(abstract[k])
                        outputfileGraph.write(str(i) + ' ' + str(len(abstractList) - 1 + 10000) + '\n')
        for j in range(len(abstractList)):
            outputfileCoauthors.write(str(j) + ' ' + abstractList[j] + '\n')
        outputfileCoauthors.close()
        outputfileGraph.close()

    def getCoauthor(namepath, data, name):

        originalData = data.split('\n\n')
        outputfileGraph = open(namepath + 'coauthorGraph.txt', 'w')
        outputfileCoauthors = open(namepath + 'coauthorList.txt', 'w')
        coauthors = []
        for i in range(0, len(originalData)):
            if originalData[i] != '':
                authorList = re.findall('author name:[^\n]+', originalData[i])
                for j in range(0, len(authorList)):
                    authorList[j] = authorList[j][12:]
                authorList.remove(name)
                for ele in authorList:
                    ele = re.sub('-', '', ele)
                    ele = ele.lower()
                    if ele in coauthors:
                        outputfileGraph.write(str(i) + ' ' + str(coauthors.index(ele) + 10000) + '\n')
                    else:
                        coauthors.append(ele)
                        outputfileGraph.write(str(i) + ' ' + str(len(coauthors) - 1 + 10000) + '\n')
        for j in range(len(coauthors)):
            outputfileCoauthors.write(str(j) + ' ' + coauthors[j] + '\n')
        outputfileCoauthors.close()
        outputfileGraph.close()

    def getInstitution(namepath, data, name):
        originalData = data.split('\n\n')
        outputfileGraph = open(namepath + 'institutionGraph.txt', 'w')
        outputfileVenues = open(namepath + 'institutionList.txt', 'w')
        institutions = []
        for i in range(0, len(originalData)):
            if originalData[i] != '':
                institution = re.search('author name:' + name + '\nauthor organization:[^\n]+',
                                        originalData[i]).group()[33 + len(name):]
                if institution in institutions:
                    outputfileGraph.write(str(i) + ' ' + str(institutions.index(institution) + 10000) + '\n')
                else:
                    institutions.append(institution)
                    outputfileGraph.write(str(i) + ' ' + str(len(institutions) - 1 + 10000) + '\n')
        for j in range(len(institutions)):
            outputfileVenues.write(str(j) + ' ' + institutions[j] + '\n')
        outputfileVenues.close()
        outputfileGraph.close()

    def getKeywords(namepath, data, name):
        stop_words = []
        print("readbefore")
        stopwordsfile = open("stopwords.txt", "r")
        print("readafter")
        stop_words_original = stopwordsfile.read().split("\n")
        for ele in stop_words_original:
            if ele != "":
                stop_words.append(ele)
        dicts = {i: '' for i in punctuation}

        def getWords(example_sent):

            example_sent = example_sent.lower()
            punc_table = str.maketrans(dicts)
            example_sent = example_sent.translate(punc_table)

            word_tokens = word_tokenize(example_sent)
            print(word_tokens)

            for i in range(len(word_tokens)):
                if word_tokens[i] in stop_words or word_tokens[i] in punctuation:
                    word_tokens[i] = "***"

            print(word_tokens)

            rst = []
            i = 0
            while i < len(word_tokens):
                if word_tokens[i] != "***":
                    rst.append(word_tokens[i])
                i += 1

            print(rst)
            return rst

        originalData = data.split('\n\n')
        outputfileGraph = open(namepath + 'keywordsGraph.txt', 'w')
        outputfileList = open(namepath + 'keywordsList.txt', 'w')

        titleWordsList = []
        for i in range(0, len(originalData)):
            if originalData[i] != '':
                title = re.search('title:[^\n]*', originalData[i]).group()[6:]
                titleWords = getWords(title)
                for word in titleWords:
                    if word in titleWordsList:
                        outputfileGraph.write(str(i) + ' ' + str(titleWordsList.index(word) + 10000) + '\n')
                    else:
                        titleWordsList.append(word)
                        outputfileGraph.write(str(i) + ' ' + str(len(titleWordsList) - 1 + 10000) + '\n')

        outputfileGraph.close()

        for i in range(len(titleWordsList)):
            outputfileList.write(str(i) + titleWordsList[i] + '\n')
        outputfileList.close()

    def getReference(namepath, data, name):

        originalData = data.split('\n\n')

        outputfileGraph = open(namepath + 'referenceGraph.txt', 'w')
        outputfileAuthors = open(namepath + 'referenceList.txt', 'w')
        referenceList = []
        for i in range(0, len(originalData)):
            if originalData[i] != '':
                references = re.search('references:[^\n]*', originalData[i]).group()[11:].split('|')
                for k in range(len(references)):
                    if references[k] in referenceList:
                        outputfileGraph.write(str(i) + ' ' + str(referenceList.index(references[k]) + 10000) + '\n')
                    else:
                        referenceList.append(references[k])
                        outputfileGraph.write(str(i) + ' ' + str(len(referenceList) - 1 + 10000) + '\n')

        for j in range(len(referenceList)):
            outputfileAuthors.write(str(j) + ' ' + referenceList[j] + '\n')
        outputfileAuthors.close()
        outputfileGraph.close()

    def getStudyfields(namepath, data, name):

        originalData = data.split('\n\n')
        outputfileGraph = open(namepath + 'studyfieldsGraph.txt', 'w')
        outputfileCoauthors = open(namepath + 'studyfieldsList.txt', 'w')

        studyfieldsList = []
        for i in range(0, len(originalData)):
            if originalData[i] != '':
                studyfields = re.search('study fields:[^\n]*', originalData[i]).group()[13:].split('|')
                for k in range(len(studyfields)):
                    studyfields[k] = studyfields[k].lower()
                    if studyfields[k] in studyfieldsList:
                        outputfileGraph.write(str(i) + ' ' + str(studyfieldsList.index(studyfields[k]) + 10000) + '\n')
                    else:
                        studyfieldsList.append(studyfields[k])
                        outputfileGraph.write(str(i) + ' ' + str(len(studyfieldsList) - 1 + 10000) + '\n')
        for j in range(len(studyfieldsList)):
            outputfileCoauthors.write(str(j) + ' ' + studyfieldsList[j] + '\n')
        outputfileCoauthors.close()
        outputfileGraph.close()

    def getVenue(namepath, data, name):

        originalData = data.split('\n\n')
        outputfileGraph = open(namepath + 'venueGraph.txt', 'w')
        outputfileVenues = open(namepath + 'venueList.txt', 'w')
        venues = []
        for i in range(0, len(originalData)):
            if originalData[i] != '':
                venue = re.search('venue name:[^\n]*', originalData[i]).group()[11:]
                if venue in venues:
                    outputfileGraph.write(str(i) + ' ' + str(venues.index(venue) + 10000) + '\n')
                else:
                    venues.append(venue)
                    outputfileGraph.write(str(i) + ' ' + str(len(venue) - 1 + 10000) + '\n')
        for j in range(len(venues)):
            outputfileVenues.write(str(j) + ' ' + venues[j] + '\n')
        outputfileVenues.close()
        outputfileGraph.close()

    getAbstract(namepath, data, name)
    getCoauthor(namepath, data, name)
    getInstitution(namepath, data, name)
    getKeywords(namepath, data, name)
    getReference(namepath, data, name)
    getStudyfields(namepath, data, name)
    getVenue(namepath, data, name)
