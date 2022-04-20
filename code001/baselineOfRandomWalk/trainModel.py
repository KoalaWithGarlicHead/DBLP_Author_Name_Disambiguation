import gensim
import random
import pickle

path="data/"

class TrainModel:

    def trainModel(self,name,weightCoauthor,weightStudyfields,weightAbstract,weightReference,p, walkLength, loopCount,
                   word2vecSize, word2vecWindow,word2vecSG, word2vecP, count, numNetwork):
        model = gensim.models.Word2Vec.load(path+'gensimModel-'+name+'-'+str(weightCoauthor)+','+str(weightStudyfields)+','+
                                            str(weightAbstract)+','+str(weightReference)+'('+str(p)+')'+'-'+str(walkLength)+
                                            ','+str(loopCount)+','+str(word2vecSize)+','+str(word2vecWindow)+','+str(word2vecSG)+'.model')
        length = len(model.wv.vocab)

        visit = []  # 用来判断元素是否被访问过
        for i in range(length):
            visit.append(-1)

        score = 1  # 同样的score代表一个作者

        # print(len(visit))

        def getMostSimilar(i, p, count):
            '''

            用在trainModel中
            :param i: 循环的index
            :param p: 相似度阀值
            :param count: 迭代次数
            :return:
            '''
            num = str(i)
            k = count - 1
            while k >= 0:
                if model.most_similar(num, topn=count)[k][1] > p:
                    break
                else:
                    k -= 1
            return k

        for i in range(length):
            num = str(i)
            k = getMostSimilar(i, word2vecP, count)
            if k != -1:
                firstVisitedBefore = 0
                while firstVisitedBefore <= k:
                    if visit[int(model.most_similar(num, topn=count)[firstVisitedBefore][0])] != -1:
                        break
                    firstVisitedBefore += 1

                if firstVisitedBefore == k + 1:
                    for j in range(k + 1):
                        visit[int(model.most_similar(num, topn=count)[j][0])] = score
                    visit[i] = score
                    score += 1

                else:
                    changescore = visit[int(model.most_similar(num, topn=count)[firstVisitedBefore][0])]
                    visit[i] = changescore

        return visit

        # # 获得对照组数据
        # contrastFile = open(
        #      name + '-rst-clean.txt',
        #     'r')
        # # contrastFile = open('/Users/littlepig/Desktop/dblp大创/词语集/' + name + '-rst-clean.txt', 'r')
        # contrastFileData = contrastFile.read().split('\n')[:-1]
        # contrast = []
        # p = 0
        # count = 0
        # for ele in contrastFileData:
        #     pCurrent = ele.split(' ')[1]
        #     if pCurrent != p:
        #         count += 1
        #         p = pCurrent
        #     contrast.append(count)
        # print(contrast)

        # f1 = open("/Users/littlepig/PycharmProjects/dblpSearch/line/"+name+"/manyArticles-"+str(numNetwork)+".pkl","rb")
        # manyArticles = pickle.load(f1)
        # f1.close()

        # print(manyArticles)

        # authorPerCount = [len(ele) for ele in manyArticles]
        # authorCountIncreased = [sum(authorPerCount[:i]) for i in range(len(authorPerCount))]

        # testVisit = []
        # for ele in manyArticles:
        #     for index in ele:
        #         testVisit.append(visit[index])

        # print(testVisit)

        # f2 = open("/Users/littlepig/PycharmProjects/dblpSearch/line/"+name+"/test_positive_combined-"+str(numNetwork)+".pkl","rb")
        # test_positive_combined = pickle.load(f2)
        # f2.close()

        # f3 = open("/Users/littlepig/PycharmProjects/dblpSearch/line/"+name+"/test_negative_combined-"+str(numNetwork)+".pkl","rb")
        # test_negative_combined = pickle.load(f3)
        # f3.close()

        # def getElementIndexInAuthorCountIncreased(num):
        #     for i in range(1, len(authorPerCount)):
        #         if num <= authorCountIncreased[i]:
        #             return i - 1
        #     return len(authorPerCount) - 1

        # print(len(testVisit))
        # print(test_positive_combined)
        # print(test_negative_combined)

        # test_positive_indexs = [[] for i in range(len(authorPerCount))]
        # test_negative_indexs = [[] for i in range(len(authorPerCount))]

        # for i in range(len(test_positive_combined)):
        #     num = test_positive_combined[i][0]
        #     index = getElementIndexInAuthorCountIncreased(num)
        #     test_positive_indexs[index].append(i)

        # for i in range(len(test_negative_combined)):
        #     numL = test_negative_combined[i][0]
        #     indexL = getElementIndexInAuthorCountIncreased(numL)
        #     test_negative_indexs[indexL].append(i)
        #     numR = test_negative_combined[i][1]
        #     indexR = getElementIndexInAuthorCountIncreased(numR)
        #     test_negative_indexs[indexR].append(i)

        # print(test_positive_indexs)

        # totalPairsToSameAuthor = 0
        # pairsCorrectlyPredictedToSameAuthor = 0
        # pairsPredictedToSameAuthor = 0

        # for i in range(len(authorPerCount)):
        #     positive_Indexs = test_positive_indexs[i]
        #     negative_Indexs = test_negative_indexs[i]

        #     totalPairsToSameAuthor += len(positive_Indexs)

        #     pairsCorrectlyPredictedToSameAuthorTEMP = 0
        #     for j in range((len(positive_Indexs))):
        #         positivePair = test_positive_combined[positive_Indexs[j]]
        #         if testVisit[positivePair[0]] == testVisit[positivePair[1]]:
        #             pairsCorrectlyPredictedToSameAuthorTEMP += 1

        #     pairsPredictedToSameAuthorTEMP = 0
        #     for j in range(len(negative_Indexs)):
        #         negativePair = test_negative_combined[negative_Indexs[j]]
        #         if testVisit[negativePair[0]] == testVisit[negativePair[1]]:
        #             pairsPredictedToSameAuthorTEMP += 1

        #     pairsPredictedToSameAuthorTEMP = pairsPredictedToSameAuthorTEMP/2+pairsCorrectlyPredictedToSameAuthorTEMP

        #     pairsCorrectlyPredictedToSameAuthor += pairsCorrectlyPredictedToSameAuthorTEMP
        #     pairsPredictedToSameAuthor += pairsPredictedToSameAuthorTEMP

        # precision = pairsCorrectlyPredictedToSameAuthor / pairsPredictedToSameAuthor
        # recall = pairsCorrectlyPredictedToSameAuthor / totalPairsToSameAuthor

        # F1 = 2*precision*recall/(precision+recall)

        # print(precision,recall,F1)







        #
        #
        # # 真正的算准确率，召回率，F值
        # precision = 0
        # recall = 0
        # lengthCompute = length
        # for i in range(length):
        #
        #     R = visit[i]
        #     RList = []
        #     if R == -1:
        #         RList.append(i)
        #     else:
        #         for j in range(length):
        #             if visit[j] == R:
        #                 RList.append(j)
        #
        #     S = contrast[i]
        #     SList = []
        #     for j in range(length):
        #         if contrast[j] == S:
        #             SList.append(j)
        #
        #     commonBetweenRListSList = []
        #     for ele in RList:
        #         if ele in SList:
        #             commonBetweenRListSList.append(ele)
        #
        #     recall += len(commonBetweenRListSList) / len(SList)
        #     precision += len(commonBetweenRListSList) / len(RList)
        #
        # precision /= lengthCompute
        # recall /= lengthCompute
        # F_measure = 2 * precision * recall / (precision + recall)
        #
        # print(name + '的precision:' + str(precision))
        # print(name + '的recall:' + str(recall))
        # print(name + '的F_measure:' + str(F_measure))
