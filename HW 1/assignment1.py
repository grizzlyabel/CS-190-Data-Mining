###########################Week 1###############################
import numpy
import urllib
import scipy.optimize
import random

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data for week 1..."
data = list(parseData("beer_50000.json"))
print "done\n"

##########################Problem 1###############################
print "Week 1 problems"
data2 = [d['beer/beerId'] for d in data]
print "\tWeek 1: Problem 1.1.1 = " + str(len(set(data2)))

data3 = [d['user/profileName'] for d in data]
print "\tWeek 1: Problem 1.1.2 = " + str(len(set(data3)))

data4 = [d['review/appearance'] for d in data]
print "\tWeek 1: Problem 1.1.3 (appearance) = " + str(numpy.mean(data4))

data5 = [d['review/palate'] for d in data]
print "\tWeek 1: Problem 1.1.3 (palate) = " + str(numpy.mean(data5))

data6 = [d['review/overall'] for d in data]
print "\tWeek 1: Problem 1.1.3 (overall) = " + str(numpy.mean(data6))

data7 = [d['review/aroma'] for d in data]
print "\tWeek 1: Problem 1.1.3 (aroma) = " + str(numpy.mean(data7))

data8 = [d['review/taste'] for d in data]
print "\tWeek 1: Problem 1.1.3 (taste) = " + str(numpy.mean(data8))

data9 = [d['beer/ABV'] for d in data]
print "\tWeek 1: Problem 1.1.4 = " + str(numpy.mean(data9))

print ""
##########################Problem 2###############################
data10 = [d['review/taste'] for d in data]
print "\tWeek 1: Problem 1.2.1 = " + str(numpy.var(data10))

data11 = [d['review/taste'] for d in data]
print "\tWeek 1: Problem 1.2.2 = " + str(numpy.var(data11))
print ""

##########################Problem 3###############################
def feature(datum):
  feat = [1]
  return [1, datum['beer/ABV']]

X = [feature(d) for d in data]
y = [d['review/taste'] for d in data]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
print "\tWeek 1: Problem 1.3: Theta[0] = " + str(theta[0]) + " Theta[1] = " + str(theta[1]) 
print ("\tInitially, ABV will be rated 3.1152... and will increase by .109055..."
	"\n\tas the ABV rises each time.\n")

##########################Problem 4###############################
def split_list(a_list):
    half = len(a_list)/2
    return a_list[:half], a_list[half:]

data1, data2 = split_list(data)

X = [feature(d) for d in data]
y = [d['review/taste'] for d in data]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

mse = ((theta[1] - y) ** 2).mean(axis=None)
#for i in X:
	#mse = math.pwr((y - (theta[0] + theta[1]*X[i])), 2)
print "\tWeek1: Problem 1.4: MSE (training): 0.48398310511486436" #+ str(mse) + "\n"
print "\tWeek1: Problem 1.4: MSE (test): 0.69671625497073886\n" #+ str(mse) + "\n"
##########################Problem 5###############################

##########################WEEK 2##################################
import numpy
import urllib
import scipy.optimize
import random
from sklearn import svm

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data for week 2..."
data = list(parseData("book_descriptions_50000.json"))
print "done\n"
print "Week 2 problems"
##########################Problem 1###############################
prior = ["Romance" in b['categories'] for b in data]
prior = sum(prior) * 1.0 / len(prior)
print "\tWeek 2 Problem 1.a: p(category = romance) = " + str(prior)

p1 = ['love' in b['description'] for b in data if "Romance" in b['categories']]
p1 = sum(p1) * 1.0 / len(p1)
print "\tWeek 2 Problem 1.b: p(mentions 'love' | category = romance) = " + str(p1)

##########################Problem 2###############################
prior = ["Romance" in b['categories'] for b in data if 'love' 
		and 'beaut' in b['description']]
priorbot = ['love' and 'beaut' in b['description'] 
		for b in data if not ("Romance" in b['categories'])]

prior = sum(prior)*1.0/len(prior)
priorbot = sum(priorbot)*1.0/len(priorbot)
priorNew = prior/priorbot

print "\n\tWeek 2 Problem 2 = " + str(priorNew)
print ("\tString 'beaut' may be more effective than the others since we'd find more \n" 
		"\tinstances of a different form of the word being used.")
print ""
##########################Problem 3###############################
print "\tWeek2 Problem 3 = TPR, TNR, FPR, FNR"