import random
from gensim.models import word2vec

path="data/"

class RW:
    m=[]
    name = ''
    suffix = ''

    def setname(self, name):
        self.name = name

    def setSuffix(self,suffix):
        self.suffix = suffix

    def setM(self):
        inputfile = open(path+self.name+self.suffix+'.txt', 'r')
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

class GetRandomWork:
    def getRW(self, name, weightCoauthor, weightStudyfields,weightAbstract, weightReference, p, walkLength, loopCount, word2vecSize, word2vecWindow, word2vecSG):
        rw = RW()
        rw.setname(name)
        rw.setSuffix('-'+str(weightCoauthor)+','+str(weightStudyfields)+','+str(weightAbstract)+','+str(weightReference)+'('+str(p)+')')
        rw.setM()

        # 采用32次循环 每次游走长度为40
        dw=rw.deepwalk(walkLength,loopCount)
        walks = []
        for i in range(loopCount):
            walks.extend(next(dw))
            yield i
        print("modeling")
        # sg = 1 用CBOW的方法
        model = word2vec.Word2Vec(walks, min_count=5, size=word2vecSize, window=word2vecWindow,sg=word2vecSG)
        model.save(path+'gensimModel-'+name+'-'+str(weightCoauthor)+','+str(weightStudyfields)+','+str(weightAbstract)+','+str(weightReference)+'('+str(p)+')'
                    +'-'+str(walkLength)+','+str(loopCount)+','+str(word2vecSize)+','+str(word2vecWindow)+','+str(word2vecSG)+'.model')
        yield -1