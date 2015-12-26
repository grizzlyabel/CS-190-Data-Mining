import numpy
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("beer_50000.json"))
print "done"

X = [[x['review/overall'], x['review/taste'], x['review/aroma'], x['review/appearance'], x['review/palate']] for x in data]

#1.1
X_array = numpy.array(X)
fiveD_mean = X_array.mean(axis=0) #when axis =0; its columns
print "\t<><>Task 1.1<><>"
print ("\tThe mean of the 5-Dimensional array is: " + str(fiveD_mean) + "\n")

#1.2
compressed_data = numpy.array((fiveD_mean - X)**2)
y = sum(compressed_data)
z = sum(y)
print "\t<><>Task 1.2<><>"
print ("\tThe 'Reconstruction Error' is: " + str(z) + "\n")


#1.3
# Initialize Centroids
centroids = [[0,0,0,0,1], [0,0,0,1,0]]
def distance(x, y):
	summed = 0
	for x1, y1 in zip(x, y):
		a = (x1-y1)**2
		summed = summed + a
	return summed
def centroidAlg(centroid, datum):
	Y = []
	centroid0Array = [centroid[0]]
	centroid1Array = [centroid[1]]
	for d in datum:
		x1 = distance(d,centroid[0])
		x2 = distance(d,centroid[1])
		if (x1 < x2):
			centroid0Array.append(d)
			Y.append(0)
		else:
			centroid1Array.append(d)
			Y.append(1)
	a = numpy.array(centroid0Array)
	b = numpy.array(centroid1Array)
	centroid[0] = a.mean(axis=0)
	centroid[1] = b.mean(axis=0)
	return [centroid,Y]
centroidReturn0 = centroidAlg(centroids, X)
centroidReturn1 = centroidAlg(centroidReturn0[0], X)
def comparison(x, y):
	for x1, y1 in zip(x,y):
		if (x1 != y1):
			return False
	return True
def repeat(centroids, datum):
	centroidReturn0 = centroidAlg(centroids,X)
	centroidReturn1 = centroidAlg(centroidReturn0[0],X)
	while True:
		if(comparison(centroidReturn0[1],centroidReturn1[1])):
			return centroidReturn0[0]
		else:
			centroidReturn0 = centroidAlg(centroidReturn0[0],X)
			if(comparison(centroidReturn1[1],centroidReturn0[1])):
				return centroidReturn1[0]
			else:
				centroidReturn1=centroidAlg(centroidReturn1[0],X)
result = repeat(centroids,X)
print "\t<><>Task 1.3<><>"
print ("\tCentroids of two clusters after convergence: \n\t\t" + str(result[0]) + "\n"
		+ "\t\t" + str(result[1]) + "\n")

#1.5
c_one = [[4.17993, 4.23675, 4.14107, 4.08866, 4.12518], [3.09862, 3.06899, 3.14020, 3.38222, 3.11332]]
def distance(x, y): #Where x is the data points and y is centroids compared to
	summed = 0
	for x1, y1 in zip(x, y):
		a = (x1-y1)**2
		summed = summed + a
	return summed

def centroidAlg(centroid, datum):
	c_one_counter = 0
	c_two_counter = 0
	Y = []
	arrayC0 = []
	centroid0Array = [c_one[0]]
	centroid1Array = [c_one[1]]
	for d in datum:
		x1 = distance(d,c_one[0])
		x2 = distance(d,c_one[1])
		if (x1 < x2):
			centroid0Array.append(d)
			Y.append(0)
			c_one_counter+=1
			arrayC0.append(c_one[0])
		else:
			centroid1Array.append(d)
			Y.append(1)
			c_two_counter+=1
			arrayC0.append(c_one[1])
	a = numpy.array(centroid0Array)
	b = numpy.array(centroid1Array)

	c_one[0] = a.mean(axis=0)
	c_one[1] = b.mean(axis=0)
	return [c_one,Y, c_one_counter, c_two_counter, arrayC0]

centroidReturn0 = centroidAlg(c_one, X)
centroidReturn1 = centroidAlg(centroidReturn0[0], X)

def comparison(x, y):
	for x1, y1 in zip(x,y):
		if (x1 != y1):
			return False
	return True

def repeat(centroids, datum):
	centroidReturn0 = centroidAlg(c_one,X)
	centroidReturn1 = centroidAlg(centroidReturn0[0],X)
	while True:
		if(comparison(centroidReturn0[1],centroidReturn1[1])):
			return centroidReturn0[0]
		else:
			centroidReturn0 = centroidAlg(centroidReturn0[0],X)
			if(comparison(centroidReturn1[1],centroidReturn0[1])):
				return centroidReturn1[0]
			else:
				centroidReturn1=centroidAlg(centroidReturn1[0],X)
result = repeat(c_one,X)

result = centroidAlg(c_one, X)
print "\t<><>Task 1.5<><>"
print "\t" + str(result[2]) + " points are closest to c1."
print "\t" + str(result[3]) + " points are closest to c2.\n"

#1.6
t1_6 = sum([sum([(c[i] - x[i]) ** 2 for i in range(5)]) for c, x in zip(result[4], X)])
print "\t<><>Task 1.6<><>"
print "\tReconstruction error is: " + str(t1_6) + "\n"

####################################Week 2####################################
import numpy
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
#2.1
edges = set()
nodes = set()
for edge in urllib.urlopen("http://jmcauley.ucsd.edu/cse255/data/facebook/egonet.txt", 'r'):
  x,y = edge.split()
  x,y = int(x),int(y)
  edges.add((x,y))
  edges.add((y,x))
  nodes.add(x)
  nodes.add(y)

print "\t<><>Task 2.1<><>"
print "\tNumber of nodes: " + str(len(nodes))
print "\tNumber of edges: " + str(len(edges)) + "\n"

#2.2
G = nx.Graph()
for e in edges:
  G.add_edge(e[0],e[1])

# returns a list of lists w/ biggest list as first element
connected_comp = list(nx.connected_components(G))
#t2_2_2 = len(connected_comp[numpy.argmax([len(c) for c in connected_comp])])

print "\t<><>Task 2.2<><>"
print "\tThere are " + str(len(connected_comp)) + " connected components."
print ("\tThere are " + str(len(connected_comp[0])) + " nodes in the largest "
		+ "connected component.\n")

#2.3
### Find all 3 and 4-cliques in the graph ###
cliques3 = set()
cliques4 = set()
for n1 in nodes:
  for n2 in nodes:
    if not ((n1,n2) in edges): continue
    for n3 in nodes:
      if not ((n1,n3) in edges): continue
      if not ((n2,n3) in edges): continue
      clique = [n1,n2,n3]
      clique.sort()
      cliques3.add(tuple(clique))
      for n4 in nodes:
        if not ((n1,n4) in edges): continue
        if not ((n2,n4) in edges): continue
        if not ((n3,n4) in edges): continue
        clique = [n1,n2,n3,n4]
        clique.sort()
        cliques4.add(tuple(clique))
c = list(nx.k_clique_communities(G, 4))
communities = len(list(c))

print "\t<><>Task 2.3<><>"
print "\tThere are " + str(communities) + " communities."
print "\tCommunity0 nodes = " + str(c[0])
print "\tCommunity1 nodes = " + str(c[1])
print "\tCommunity2 nodes = " + str(c[2])
print "\tCommunity3 nodes = " + str(c[3])
print "\tCommunity4 nodes = " + str(c[4])
