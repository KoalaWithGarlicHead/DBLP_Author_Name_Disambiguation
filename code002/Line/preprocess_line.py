from code002.ge import LINE
import networkx as nx
import time

def attributeDoLine(name, attribute):
    namepath = "data/"+name+"/"
    G = nx.read_edgelist(
        namepath+attribute+'Graph.txt',create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = LINE(G, embedding_size=128, order='all')
    model.train(batch_size=1024, epochs=10, verbose=2)
    embeddings = model.get_embeddings()
    file = open(namepath+'rst-'+attribute, 'w')
    count = 0
    for i in embeddings:
        if int(i) < 10000:
            count += 1
            file.write(str(i) + '\n')
            file.write(str(embeddings[str(i)]) + '\n\n')
    file.close()

def doLine(name):
    attributeDoLine(name, 'abstract')
    time.sleep(2)
    attributeDoLine(name, 'coauthor')
    attributeDoLine(name, 'institution')
    attributeDoLine(name, 'keywords')
    attributeDoLine(name, 'reference')
    attributeDoLine(name, 'venue')
    attributeDoLine(name, 'studyfields')

