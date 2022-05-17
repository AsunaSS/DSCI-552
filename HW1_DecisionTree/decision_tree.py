'''
Chaoyu Li
chaoyuli@usc.edu
2/2/2022
'''

import math
import matplotlib.pyplot as plt
import operator

#Calculate the Entopy
def calcEnt(dataSet):
    ent = len(dataSet)
    #create dict for all labels
    labels = {}
    for feature in dataSet:
        temp = feature[-1]
        if temp not in labels.keys():
            labels[temp] = 0
        labels[temp] += 1
    #caculate Entropy
    Ent = 0.0
    for key in labels:
        prob = float(labels[key]) / ent
        Ent -= prob * math.log(prob, 2)
    return Ent

#Split by a feature
def splitDataSet(dataSet,feature,value):
    dataSet_list = []
    for featureVector in dataSet:
        if featureVector[feature] == value:
            temp = featureVector[:feature]
            temp.extend(featureVector[feature+1:])
            dataSet_list.append(temp)
    return dataSet_list #Returns a subset without split features

#Split data by maximum information gain
def chooseBestFeatureToSplit(dataSet):
    n = len(dataSet[0]) - 1
    deEnt = calcEnt(dataSet)
    bestI = 0.0
    bestFeature = -1
    for i in range(n):
        feature_list = [number[i] for number in dataSet] #Get all values under a feature (a column)
        unique = set(feature_list) #make sure that no duplicate feature value
        #print(unique)
        newEnt = 0.0
        for value in unique:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet)) #p(t)
            newEnt += prob * calcEnt(subDataSet)
        infoGain = deEnt - newEnt #caculate Information Gain
        #print(infoGain)
        #print()
        #find the biggest Information Gain
        if (infoGain > bestI):
            bestI = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(feature):
    class_dict = {}
    for i in feature:
        if i not in class_dict.keys():
            class_dict[i] = 0
        class_dict[i] = class_dict[i] + 1
    class_list = sorted(class_dict.items,key=operator.itemgetter(1),reversed=True)
    return class_list[0][0]

def createTree(dataSet,labels):
    feature = [example[-1] for example in dataSet]
    #If the class is the same, return
    #print(feature)
    if feature.count(feature[0]) == len(feature):
        return feature[0]
    #If the length of dataset[0] == 1, return the feature shown most
    if len(dataSet[0]) == 1:
        return majorityCnt(feature)
    #Choose the best feature based on the informationGain
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestLabel = labels[bestFeat]
    #print("!@#!@!")
    #print(bestLabel)
    #print()
    dTree = {bestLabel:{}}
    del (labels[bestFeat])
    feature_list = [example[bestFeat] for example in dataSet]
    unique = set(feature_list)
    for value in unique:
        subLabels = labels[:]
        #print("!!!!!!!!")
        #print(bestLabel, value)
        #print(subLabels)
        #print("!!!!!!!!")
        dTree[bestLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return dTree

#Setting the format of node in the plot
def plotNode(string,centerPt,upPos, type, size):
    createPlot.ax1.annotate(string,xy=upPos,xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=type,arrowprops=dict(arrowstyle='<-'),fontsize=size)

#dfs to decide the width of the plot
def getNumLeafs(dTree):
    numLeafs = 0
    temp1 = list(dTree.keys())[0]
    temp2 = dTree[temp1]
    for key in temp2.keys():
        if type(temp2[key]).__name__=='dict':
            numLeafs += getNumLeafs(temp2[key])
        else:
            numLeafs += 1
    return numLeafs

#dfs to decide the height of the plot
def getTreeDepth(dTree):
    depth = 0
    temp1 = list(dTree.keys())[0]
    temp2 = dTree[temp1]
    for key in temp2.keys():
        if type(temp2[key]).__name__=='dict':
            depthTemp = 1 + getTreeDepth(temp2[key])
        else: 
            depthTemp = 1
        if depthTemp > depth:
            depth = depthTemp
    return depth

#find the position to show the text between parent nodes and child nodes
def plotTextPosition(cntPos, upPos, string):
    x = (upPos[0] - cntPos[0]) / 9 + 0.995 * cntPos[0]
    y = (upPos[1] - cntPos[1]) / 4.5 + cntPos[1]
    createPlot.ax1.text(x, y, string, fontsize=8)

#Draw the nodes and text information of the decision tree
def plotTree(dTree, upPos, string):
    numLeafs = getNumLeafs(dTree)
    depth = getTreeDepth(dTree)
    temp1 = list(dTree.keys())[0]
    cntPos = (plotTree.xOff + (0.6 + float(numLeafs)) / 2 / plotTree.totalW, plotTree.yOff)
    plotTextPosition(cntPos, upPos, string)
    plotNode(temp1, cntPos, upPos, decisionNode, 12)
    temp2 = dTree[temp1]
    plotTree.yOff -= 1 / plotTree.totalD
    for key in temp2.keys():
        if type(temp2[key]).__name__=='dict':
            plotTree(temp2[key], cntPos, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(temp2[key], (plotTree.xOff, plotTree.yOff), cntPos, leafNode, 10)
            plotTextPosition((plotTree.xOff, plotTree.yOff), cntPos, str(key))
    plotTree.yOff = plotTree.yOff + 1 / plotTree.totalD

#Main function to draw the decision tree
def createPlot(dTree):
    fig=plt.figure(1,facecolor='white', figsize=(19.20, 10.80))
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(dTree))-3.9
    plotTree.totalD = float(getTreeDepth(dTree))
    plotTree.xOff = -2.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(dTree, (0.5, 1.0), '')
    plt.savefig("Decision-Tree.png")
    plt.show()

#Load txt file to get the dataset
def loadTXT(file):
    with open(file, "r") as f:
        f_line = f.readlines()
        line = f_line[0].strip('\n')
        line_list = line[1:-1].split(", ")
        Header = []
        for str in line_list:
            Header.append(str)
        result = []
        for line in f_line[2:]:
            line = line.strip('\n')
            line_list = line[4:-1].split(", ")
            result.append(line_list)
    return Header, result

#Give a prediction based on the query given
def predictor(dTree, query):
    if type(dTree) == dict:
        node = list(dTree.keys())[0]
        if node in query:
            return predictor(dTree[node][query[node]], query)
        else:
            return node
    else:
        return dTree

if __name__ == "__main__":
    #Open and load dataset
    dcHeadings, data = loadTXT('dt_data.txt')
    dcHeadings = dcHeadings[:-1]
    print(data)
    print(dcHeadings)

    #Build the decision tree based on the dataset
    tree = createTree(data, dcHeadings)
    print(tree)

    #Set the format and style of the nodes of the decision tree
    decisionNode = dict(boxstyle='roundtooth',fc='0.5',pad=2) #style of decision node(feature)
    leafNode = dict(boxstyle='round4',fc='0.8',pad=1.2) #style of leaf node(yes/no)

    #Draw the decision tree based on the Tree dict
    createPlot(tree)

    #Give a prediction based on the conditions given
    query = {'Occupied':'Moderate','Price':'Cheap','Music':'Loud','Location':'City-Center','VIP':'No','Favorite Beer':'No'}
    print(predictor(tree, query))