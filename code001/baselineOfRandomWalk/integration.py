import gensim
from code001.baselineOfRandomWalk.getSimilarities import GetSimilarities
from code001.baselineOfRandomWalk.doRandomWalk import GetRandomWork
from code001.baselineOfRandomWalk.trainModel import TrainModel


# from getSimilarities import GetSimilarities
# from doRandomWalk import GetRandomWork
# from trainModel import TrainModel

def disambiguation(data, name, numNetwork, weightCoauthor=4, weightStudyfields=2.5, weightAbstract=2.5,
                   weightReference=1, p=0.1,
                   walkLength=40, loopCount=32, word2vecSize=10, word2vecWindow=3, word2vecSG=1, word2vecP=0.9,
                   count=8):
    similarities = GetSimilarities()
    simCor = similarities.getSimilarities(data, name, weightCoauthor, weightStudyfields, weightAbstract,
                                          weightReference, p)
    total = next(simCor) + 32
    i = 0
    while True:
        x = next(simCor)
        if x == -1:
            break
        print(x)
        yield int(i * 100.0 / total)
        i += 1

    print("*********")

    rw = GetRandomWork()
    rwCor = rw.getRW(name, weightCoauthor, weightStudyfields, weightAbstract, weightReference, p, walkLength, loopCount,
                     word2vecSize, word2vecWindow, word2vecSG)
    while True:
        x = next(rwCor)
        if x == -1:
            break
        print(x)
        yield int(i * 100.0 / total)
        i += 1

    print("*********")
    print("training")
    tm = TrainModel()
    rst = tm.trainModel(name, weightCoauthor, weightStudyfields, weightAbstract, weightReference, p, walkLength,
                        loopCount, word2vecSize, word2vecWindow, word2vecSG, word2vecP, count, numNetwork)

    yield -1
    yield rst


if __name__ == "__main__":
    with open("../../Wei Li-unmarked-clean2.txt", "r", encoding="utf-8") as f:
        cor = disambiguation(f.read(), 'Wei Li', 20, word2vecSize=20, word2vecWindow=3, word2vecP=0.75)
        while True:
            rate = next(cor)
            if rate == -1:
                print("*********")
                print("done")
                break
