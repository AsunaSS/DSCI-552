'''
Chaoyu Li
Date: 2/18/2022
'''
import copy
import numpy as np
import matplotlib.pyplot as plt

dimension = 2
numPoints = 10

# Get two furthest pivot points
def findFarthestPoints(disMatrix):
    numPoints = len(disMatrix)
    # Initialize the first pivot with a random number
    first = np.random.randint(0, numPoints)
    while True:
        second = np.argmax(disMatrix[first])
        temp = np.argmax(disMatrix[second])
        if temp == first:
            break
        else:
            first = second
    # Return a tuple
    result = (min(first, second), max(first, second))
    return result

# Calculate the projecting coordinate with the formula xi = (Dai^2 + Dab^2 - Dbi^2)/(2*Dab)
def projection(disMatrix, pair, point):
    xi = (disMatrix[pair[0]][point] ** 2 + disMatrix[pair[0]][pair[1]] ** 2 -
          disMatrix[pair[1]][point] ** 2) / (2 * disMatrix[pair[0]][pair[1]])
    return xi

# Update the new distance matrix with the formula D'(Oi, Oj) = sqrt(D(Oi, Oj)^2 - (xi-xj)^2)
def updateDisMatrix(disMatrix, X):
    result = copy.deepcopy(disMatrix)
    numPoints = len(disMatrix)
    for i in range(numPoints):
        for j in range(numPoints):
            result[i][j] = np.sqrt(disMatrix[i][j] ** 2 - (X[i] - X[j]) ** 2)
    return result

def fastMap(disMatrix, dimension):
    result = [[] for i in disMatrix]
    for k in range(dimension):
        first, second = findFarthestPoints(disMatrix)
        for index in range(len(disMatrix)):
            if index == first:
                dis = 0
            elif index == second:
                dis = disMatrix[first][second]
            else:
                dis = projection(disMatrix, (first, second), index)
            result[index].append(dis)
        temp = []
        for point in result:
            temp.append(point[-1])

        # Update distance matrix
        disMatrix = updateDisMatrix(disMatrix, temp)

    return result, disMatrix

def plotFastMap2D(result, word_list):
    result = np.asarray(result)
    fig, ax = plt.subplots()
    ax.scatter(result[:, 0], result[:, 1], c='y')
    for index, word in enumerate(word_list):
        ax.annotate(word, (result[index]))
    plt.savefig('FastMap.png')
    plt.close()

if __name__ == '__main__':
    # Read fastMap data
    disMatrix = np.zeros((numPoints, numPoints))
    with open("fastmap-data.txt") as file:
        for line in file:
            first, second, dis = line.split()
            disMatrix[int(first) - 1][int(second) - 1] = float(dis)
            disMatrix[int(second) - 1][int(first) - 1] = float(dis)
    #print(disMatrix)

    # Fastmap
    result, disMatrix = fastMap(disMatrix, dimension)
    print("Distance Matrix:\n", disMatrix)
    print("2D result:\n", result)

    # Read word list
    word_list = []
    with open("fastmap-wordlist.txt") as file2:
        word_list = list(map(str.strip, file2.readlines()))

    for word, point in zip(word_list, result):
        print('"{0}":{1}'.format(word, point))

    # Plot words in 2D space
    plotFastMap2D(result, word_list)