import os
import re
from gensim.corpora import Dictionary
import gensim
import random
from gensim.models import word2vec

path = "data/"


class Utils:

    def getSimilarities(self, data, name):
        print("this is get similarities")
        p = 0.1
        if not os.path.exists(path):
            os.mkdir(path)
        namepath = path+name+"/"
        if not os.path.exists(namepath):
            os.mkdir(namepath)
        outputfileAuthor = open(namepath + name + '-Author.txt', 'w')
        outputfileAbstact = open(namepath + name + '-Abstract.txt', 'w')
        outputfileStudyfields = open(namepath + name + '-Studyfields.txt', 'w')
        outputfileReference = open(namepath + name + '-Reference.txt', 'w')
        outputfileInstitution = open(namepath + name + '-Institution.txt', 'w')
        outputfileKeywords = open(namepath + name + '-Keywords.txt', 'w')
        outputfileVenue = open(namepath + name + '-Venue.txt', 'w')
        originalData = data.split('\n\n')
        elementsID = []
        elementsTitle = []
        elementsAuthor = []
        elementsAbstract = []
        elementsStudyfields = []
        elementsReference = []
        elementsInstitution = []
        elementsKeywords = []
        elementsVenue = []
        for i in range(0, len(originalData)):
            if originalData[i] != '':
                elementsID.append(re.search('id:[0-9]+\n', originalData[i]).group()[3:-1])
                elementsTitle.append(re.search('title:[^\n]+', originalData[i]).group()[6:])
                authorList = re.findall('author name:[^\n]+', originalData[i])
                for j in range(0, len(authorList)):
                    authorList[j] = authorList[j][12:]
                authorList.remove(name)
                elementsAuthor.append(authorList)
                abstractContent = re.search('abstract:[^\n]*', originalData[i]).group()[9:].split('|')
                elementsAbstract.append(abstractContent)
                studyfieldsList = re.search('study fields:[^\n]*', originalData[i]).group()[13:].split('|')
                elementsStudyfields.append(studyfieldsList)
                referenceList = re.search('references:[^\n]*', originalData[i]).group()[11:].split('|')
                elementsReference.append(referenceList)
                institutionList = re.search('author name:'+name+'\nauthor organization:[^\n]*', originalData[i]).group()[33+len(name):].split('|')
                elementsInstitution.append(institutionList)
                keywordsList = re.search('title:[^\n]*', originalData[i]).group()[6:].split('|')
                elementsKeywords.append(keywordsList)
                venueList = re.search('venue name:[^\n]*', originalData[i]).group()[11:].split('|')
                elementsVenue.append(venueList)

        def countSimilarites(elements, i):
            dictionary = Dictionary(elements)
            num_features = len(dictionary.token2id)
            corpus = [dictionary.doc2bow(ele) for ele in elements]
            kw_vector = dictionary.doc2bow(elements[i])
            # 4、创建【TF-IDF模型】，传入【语料库】来训练
            tfidf = gensim.models.TfidfModel(corpus)
            # 5、用训练好的【TF-IDF模型】处理【被检索文本】和【搜索词】
            tf_texts = tfidf[corpus]  # 此处将【语料库】用作【被检索文本】
            tf_kw = tfidf[kw_vector]
            # 6、相似度计算
            sparse_matrix = gensim.similarities.SparseMatrixSimilarity(tf_texts, num_features)
            similarities = sparse_matrix.get_similarities(tf_kw)
            similarList = []
            for e, s in enumerate(similarities, 1):
                # 重要的参数！相似度阀值取多少和最后的结果关系相差很大！
                if s < p or s > 0.99:
                    s = 0
                similarList.append([e - 1, s])
            return similarList

        count = len(elementsID)
        for i in range(0, count):

            def writeOneAttribute(elements, writefile):

                similarSingleUnit = [0 for m in range(0, count)]
                similarities = countSimilarites(elements, i)
                for j in range(0, count):
                    similarSingleUnit[j] = similarities[j][1]
                similarSingleUnit[i] = -1
                writefile.write(str(similarSingleUnit) + '\n')

            writeOneAttribute(elementsAuthor, outputfileAuthor)
            writeOneAttribute(elementsVenue, outputfileVenue)
            writeOneAttribute(elementsKeywords, outputfileKeywords)
            writeOneAttribute(elementsInstitution, outputfileInstitution)
            writeOneAttribute(elementsAbstract, outputfileAbstact)
            writeOneAttribute(elementsReference, outputfileReference)
            writeOneAttribute(elementsStudyfields, outputfileStudyfields)

        outputfileAuthor.close()
        outputfileVenue.close()
        outputfileKeywords.close()
        outputfileInstitution.close()
        outputfileAbstact.close()
        outputfileReference.close()
        outputfileStudyfields.close()
        return 0

class RW:
    m=[]
    name = ''
    suffix = ''

    def setname(self, name):
        self.name = name

    def setSuffix(self,suffix):
        self.suffix = suffix

    def setM(self):
        inputfile = open(path+self.name+"/"+self.name+self.suffix+'.txt', 'r')
        matrix = []
        for line in inputfile:
            infoStr = line[1:-2].split(',')
            info = []
            for ele in infoStr:
                info.append(float(ele))
            matrix.append(info)
        inputfile.close()
        count = len(matrix)
        self.m = [[0] * count for n in range(count)]
        for i in range(0, count):
            sumOfSimilarities = 0
            for j in range(0, count):
                sumOfSimilarities += matrix[i][j]
            sumOfSimilarities += 1  # 消除本体为-1带来的影响
            for j in range(0, count):
                if matrix[i][j] > 0:
                    self.m[i][j] = matrix[i][j] / sumOfSimilarities
            self.m[i][i] = 0  # 将自己的概率设为0

    def walkOnce(self,length,start):
        walk = [str(start)]
        cur = start
        while len(walk) < length:
            cur = number_of_certain_probability(self.m[cur])
            walk.append(str(cur))
        return walk

    def deepwalk(self,length,count):
        for i in range(count):
            walks = []
            for j in range(0, len(self.m)):
                walks.append(self.walkOnce(length,j))
                walks.append(self.walkOnce(length,j))
            yield walks



def number_of_certain_probability(probability):
    count = len(probability)
    sequence = []
    for n in range(0, count):
        sequence.append(n)
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


class Utils2:

    def getRW(self, name):
        def doRW(attribute, walkLength=40, loopCount=32, word2vecSize=10, word2vecWindow=3, word2vecSG=1):
            rw = RW()
            rw.setname(name)
            rw.setSuffix('-' + attribute)
            rw.setM()

            # 采用32次循环 每次游走长度为40
            dw = rw.deepwalk(walkLength, loopCount)
            walks = []
            for i in range(loopCount):
                walks.extend(next(dw))
            print("modeling")
            # sg = 1 用CBOW的方法
            namepath = path+name+"/"
            model = word2vec.Word2Vec(walks, min_count=5, size=word2vecSize, window=word2vecWindow, sg=word2vecSG)
            model.save(namepath+name + 'gensimModel-'+attribute + '.model')

        doRW("Author")
        doRW("Keywords")
        doRW("Studyfields")
        doRW("Abstract")
        doRW("Reference")
        doRW("Institution")
        doRW("Venue")

