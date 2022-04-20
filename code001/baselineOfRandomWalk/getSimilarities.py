import gensim
import re
import os
from gensim.corpora import Dictionary

path="data/"

class GetSimilarities():

    def getSimilarities(self, data, name, weightCoauthor, weightStudyfields,weightAbstract, weightReference, p):
        if not os.path.exists(path):
            os.mkdir(path)
        outputfile = open(path+name+'-'+str(weightCoauthor)+','+str(weightStudyfields)+','+str(weightAbstract)+','+str(weightReference)+'('+str(p)+').txt', 'w')
        originalData = data.split('\n\n')
        elementsID = []
        elementsTitle = []
        elementsAuthor = []
        elementsAbstract = []
        elementsStudyfields = []
        elementsReference = []
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
                if s<p or s>0.99:
                    s = 0
                similarList.append([e-1,s])
            return similarList


        count = len(elementsID)
        matrix = []
        yield count
        for i in range(0, count):

            similarSingleUnit = [0 for m in range(0, count)]
            authorSimilarities = countSimilarites(elementsAuthor,i)
            studyfieldsSimilarities = countSimilarites(elementsStudyfields,i)
            abstractSimilarities = countSimilarites(elementsAbstract,i)
            refrenceSimilarities = countSimilarites(elementsReference,i)
            for j in range(0, count):
                similarSingleUnit[j] = authorSimilarities[j][1]*weightCoauthor+studyfieldsSimilarities[j][1]*weightStudyfields+abstractSimilarities[j][1]*weightAbstract+refrenceSimilarities[j][1]*weightReference
            similarSingleUnit[i] = -1
            matrix.append(similarSingleUnit)
            outputfile.write(str(similarSingleUnit)+'\n')
            yield i

        outputfile.close()
        yield -1
