import re
from collections import Counter
import torch
import random
import numpy as np
from itertools import combinations
from torch import nn
import pickle
import os
from nltk.tokenize import word_tokenize
import code002.Line.preprocess_line
import code002.Line.graph

from string import punctuation

path = "/Users/zhuchenqing/Desktop/dblp大创/测试数据/"
rst = ''
total = 14

def random_int_list(start, stop, length):
    random_list = []
    s = set()
    i = 0
    while i < length:
        rand = random.randint(start, stop)
        if rand not in s:
            s.add(rand)
            random_list.append(rand)
            i += 1
    return random_list


def getWhichSelect(size, num):
    rand_vector_select = random_int_list(0, size - 1, num)
    # 对任意一个Vector，生成了0.75*vector长度个数的随机数
    rand_vector_not_select = [i for i in range(size) if i not in rand_vector_select]
    return [rand_vector_select, rand_vector_not_select]


def caiyang(vector_people, ls, size, num):
    res1 = []
    res2 = []
    rand_vector_select = random_int_list(0, size - 1, num)
    # 对任意一个Vector，生成了0.75*vector长度个数的随机数

    rand_vector_not_select = [i for i in range(size) if i not in rand_vector_select]

    rand_vector_select_combined = []
    rand_vector_select_not_combined = []
    for i in rand_vector_select:
        rand_vector_select_combined.append([ls[i][0], ls[i][1]])
        l = vector_people[ls[i][0]]
        r = vector_people[ls[i][1]]
        res1.append(np.hstack((l, r)))
    for i in rand_vector_not_select:
        rand_vector_select_not_combined.append([ls[i][0], ls[i][1]])
        l = vector_people[ls[i][0]]
        r = vector_people[ls[i][1]]
        res2.append(np.hstack((l, r)))
    return [np.array(res1), np.array(res2),
            rand_vector_select_combined, rand_vector_select_not_combined]


def getContrast(contrast_data):

    contrastFileData = contrast_data.split('\n')[:-1]
    contrast = []
    p = 0
    count = 0
    for ele in contrastFileData:
        pCurrent = ele.split(' ')[1]
        if pCurrent != p:
            count += 1
            p = pCurrent
        contrast.append(count)
    return contrast


def getWeiIndexs(contrast, num):
    rst = []
    result = dict(Counter(contrast))
    for key, value in result.items():
        if value >= num:
            rst.append(key)
    return rst


def getArticleIndexs(nums, contrast):
    rstAll = []
    for num in nums:
        rst = []
        for i in range(len(contrast)):
            if contrast[i] == num:
                rst.append(i)
        rstAll.append(rst)
    return rstAll


def getSingleVector(filename):
    inputfile = open(filename, 'r')
    originalData = inputfile.read().split('\n\n')
    inputfile.close()
    vectorList = []
    k = 0
    for ele in originalData:
        if ele != '':
            k += 1
            a = re.search('\[[^\]]+\]', ele).group()[1:-1].split()
            for i in range(len(a)):
                a[i] = float(a[i])
                # a[i] = int(a[i]*10000)/10000
            vectorList.append(a)
    return vectorList


def getTotalVector(name):
    vectorCoathor = getSingleVector("data/"+name+'/rst-coauthor')
    vectorAbstract = getSingleVector("data/"+name + '/rst-abstract')
    vectorKeywords = getSingleVector("data/"+name + '/rst-keywords')
    vectorReference = getSingleVector("data/"+name + '/rst-reference')
    vectorStudyfields = getSingleVector("data/"+name + '/rst-studyfields')
    vectorVenue = getSingleVector("data/"+name + '/rst-venue')
    vectorInstitution = getSingleVector("data/"+name + '/rst-institution')
    totalVectors = []
    length = len(vectorCoathor)
    for i in range(length):
        temp = vectorCoathor[i] + vectorAbstract[i] + vectorKeywords[i] + \
               vectorReference[i] + vectorStudyfields[i] + vectorVenue[i] + vectorInstitution[i]
        totalVectors.append(temp)
    return totalVectors


def getSamples(data, name, num, contrast_data):

    code002.Line.graph.getAllAttributes(data, name)
    print("get attributes, done")

    code002.Line.preprocess_line.doLine(name)
    print("line preprocess, done")

    global rst, total
    totalVectors = getTotalVector(name)
    print("getTotalVectors, done")
    contrast = getContrast(contrast_data)
    manyWeis = getWeiIndexs(contrast, num)
    # print(manyWeis)
    length = len(manyWeis)
    # print(length)
    manyArticles = getArticleIndexs(manyWeis, contrast)
    f = open("data/"+name + '/manyArticles-' + str(num) + '.pkl', 'wb')
    pickle.dump(manyArticles, f)
    f.close()
    vectors = []
    for i in range(len(manyArticles)):
        vector = []
        for ele in manyArticles[i]:
            vector.append(totalVectors[ele])
        vectors.append(vector)

    authorPerCount = [len(vector) for vector in vectors]
    authorCountIncreased = [sum([len(vector) for vector in vectors][:i])
                            for i in range(len(vectors))]

    permus = [list(combinations(list(range(len(vector))), 2)) for vector in vectors]
    size_permus = [len(permu) for permu in permus]

    t = []
    for i in range(length):
        tu = (vectors[i], permus[i], size_permus[i])
        t.append(tu)

    train_positive_vec = []
    test_positive_vec = []
    print(1)
    yield int(1 * 100.0 / total)

    test_positive_combined_indexs = [[] for _ in range(length)]
    test_positive_combined = []  # 每一个元素包含所有选出来的组合数
    i = 0
    pianyi = 0
    testNum = []
    for tu in t:
        testNum.append(tu[2] - int(tu[2] * 0.75))
        tp = caiyang(tu[0], tu[1], tu[2], int(tu[2] * 0.75))

        rand_vector_test_combined = tp[3]
        for j in range(len(rand_vector_test_combined)):
            ele = rand_vector_test_combined[j]
            ele[0] += authorCountIncreased[i]
            ele[1] += authorCountIncreased[i]
            test_positive_combined.append(ele)
            test_positive_combined_indexs[i].append(j + pianyi)

        pianyi = len(test_positive_combined)

        if len(train_positive_vec) == 0:
            train_positive_vec = tp[0]
            test_positive_vec = tp[1]
        else:
            train_positive_vec = np.vstack((train_positive_vec, tp[0]))
            test_positive_vec = np.vstack((test_positive_vec, tp[1]))

        i += 1

    train_positive_size = train_positive_vec.shape[0]  # 训练集中正例个数
    test_positive_size = test_positive_vec.shape[0]

    sum_negative_vector = []
    sum_negative_combined = []
    for i in range(len(t) - 1):
        lv = t[i][0]
        for j in range(i + 1, len(t)):
            rv = t[j][0]
            for k in range(len(lv)):
                for l in range(len(rv)):
                    sum_negative_vector.append(np.hstack((lv[k], rv[l])))
                    sum_negative_combined.append([authorCountIncreased[i] + k,
                                                  authorCountIncreased[j] + l])

    sum_negative_vector = np.array(sum_negative_vector)
    # print(sum_negative_vector.shape)  # (11090, 3584)
    negative_select = random_int_list(0, sum_negative_vector.shape[0] - 1, train_positive_size)

    test_negative_combined_indexs = [[] for i in range(length)]

    negative_not_select = [i for i in range(sum_negative_vector.shape[0])
                           if i not in negative_select]
    train_negative_vec = sum_negative_vector[negative_select, :]
    test_negative_vec = sum_negative_vector[negative_not_select, :]

    test_negative_combined = []
    for ele in negative_not_select:
        test_negative_combined.append(sum_negative_combined[ele])

    def getElementIndexInAuthorCountIncreased(num):
        for i in range(1, length):
            if num <= authorCountIncreased[i]:
                return i - 1
        return length - 1

    for i in range(len(test_negative_combined)):
        ele = test_negative_combined[i]
        left = ele[0]
        right = ele[1]
        leftIndex = getElementIndexInAuthorCountIncreased(left)
        rightIndex = getElementIndexInAuthorCountIncreased(right)
        test_negative_combined_indexs[leftIndex].append(i)
        test_negative_combined_indexs[rightIndex].append(i)

    print(2)
    yield int(2 * 100.0 / total)

    train_vec = []
    train_target = []
    train_random_index_positive = [i for i in range(len(train_positive_vec))]
    train_random_index_negative = [i for i in range(len(train_positive_vec))]
    random.shuffle(train_random_index_positive)
    random.shuffle(train_random_index_negative)

    for i in range(len(train_positive_vec)):
        train_vec.append(train_positive_vec[train_random_index_positive[i]])
        train_vec.append(train_negative_vec[train_random_index_negative[i]])
        train_target.append(1)
        train_target.append(0)

    test_vec = []
    test_target = []
    for i in range(len(test_positive_vec)):
        test_vec.append(test_positive_vec[i])
        test_target.append(1)
    for i in range(len(test_negative_vec)):
        test_vec.append(test_negative_vec[i])
        test_target.append(0)

    print(3)
    yield int(3 * 100.0 / total)

    train_vec = np.array(train_vec)
    train_target = np.array(train_target)
    test_target = np.array(test_target)
    test_vec = np.array(test_vec)

    f1 = open("data/"+name + "/test_positive_combined-" + str(num) + ".pkl", "wb")
    pickle.dump(test_positive_combined, f1)
    f1.close()
    f2 = open("data/"+name + "/test_negative_combined-" + str(num) + ".pkl", "wb")
    pickle.dump(test_negative_combined, f2)
    f2.close()

    print(4)

    yield int(4 * 100.0 / total)

    num_inputs = train_vec.shape[1]
    train_size = train_vec.shape[0]
    num_outputs = 2

    model = torch.nn.Sequential(
        torch.nn.Linear(num_inputs, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, num_outputs),
    )

    num_epochs = 3
    bs = 10  # 批量大小
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for i in range((train_size - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            X = torch.tensor(train_vec[start_i: end_i], dtype=torch.float32)
            y = torch.tensor(train_target[start_i: end_i]).long()

            y_hat = model(X)
            l = criterion(y_hat, y).sum()
            # 梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        print('epoch %d, loss %.4f, train acc %.3f'
              % (epoch + 1, train_l_sum / train_size, train_acc_sum / train_size))
        yield int((epoch + 1 + 4) * 100.0 / total)

    test_size = test_vec.shape[0]
    test_vector = torch.tensor(test_vec, dtype=torch.float32)

    false = 0
    preds = []
    for i in range(test_size):
        output = int(model(test_vector[i]).argmax().item())
        preds.append(output)
        if output != int(test_target[i]):
            false += 1
    print(false / test_size)

    # fileRst = open('result.txt', 'w')
    # fileRst.write(name + ' ' + str(num) + '\n')
    rst += "author name: " + name + '\n'


    def my_f1_all():
        global rst
        test_positive_indexs_all = []
        for ele in test_positive_combined_indexs:
            for index in ele:
                test_positive_indexs_all.append(index)

        test_negative_indexs_all = []
        for ele in test_negative_combined_indexs:
            for index in ele:
                test_negative_indexs_all.append(index)

        predsPositive = preds[:len(test_positive_vec)]
        predsNegative = preds[len(test_positive_vec):]

        totalPairsToSameAuthor = len(test_positive_combined)

        pairsCorrectlyPredictedToSameAuthor = 0
        for index in test_positive_indexs_all:
            if predsPositive[index] == 1:
                pairsCorrectlyPredictedToSameAuthor += 1

        totalPairsPredictedToSameAuthor = pairsCorrectlyPredictedToSameAuthor
        for index in test_negative_indexs_all:
            if predsNegative[index] == 1:
                totalPairsPredictedToSameAuthor += 1

        precision = pairsCorrectlyPredictedToSameAuthor / totalPairsPredictedToSameAuthor
        recall = pairsCorrectlyPredictedToSameAuthor / totalPairsToSameAuthor

        f1 = 2 * precision * recall / (precision + recall)

        rst += "Precision = " + str(precision) + "\n"
        rst += "Recall = " + str(recall) + "\n"
        rst += "F1 = " + str(f1) + "\n"


    my_f1_all()

    yield -1
    yield rst

