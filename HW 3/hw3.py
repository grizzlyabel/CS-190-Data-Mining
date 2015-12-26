import gzip
import numpy
import urllib
import math
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
	
def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

data = list(parseData("train.json"))
	
##################################Problem 1#################################
#Fit a simple predictor
X = [[l['reviewerID'],l['itemID'],l['rating']] for l in data]
#Get list of reviews
y = [l[2] for l in X]
#Change to array to manipulate functions
y = numpy.array(y)
#Get mean of reviews
ymean = y.mean()  ##ANSWER PART 1
print "\n<><><>PROBLEM 1<><><>"
print "\tAlpha is: " + str(ymean)

real = []
full = []
for l in open("labeled_Rating.txt"):
	u,i,r = l.strip().split(' ')
	real.append(float(r))

for l in data:
	u,i,r = l['reviewerID'],l['itemID'],l['rating']
	full.append((u,i,r))

real = numpy.array(real)

#MSE
mse_data = numpy.mean(numpy.array((ymean - real)**2))
#mse = sum(mse_data)  ##ANSWER PART 2
print "\tMSE is: " + str(mse_data) + "\n"
##################################Problem 2#################################
print "\n<><><>Problem 2<><><>"
# Fit a predictor of the form
# 	rating(user, item) = a + B_user + B_item
# (with the regularization parameter lambda = 1)
def calc_error_reg(a, B_u, B_i, l, data):
    return sum([(a + B_u[x['reviewerID']] + B_i[x['itemID']] - x['rating']) ** 2 for x in data]) + l * (sum([x ** 2 for x in B_u.values()]) + sum([x ** 2 for x in B_i.values()]))
user_list = set([x['reviewerID'] for x in data])
item_list = set([x['itemID'] for x in data])
I_u = dict([(u, []) for u in user_list])
U_i = dict([(i, []) for i in item_list])
for x in data:
    I_u[x['reviewerID']].append({'ID': x['itemID'], 'rating': x['rating']})
    U_i[x['itemID']].append({'ID': x['reviewerID'], 'rating': x['rating']})
a = 0
B_u = dict([(u, 0) for u in user_list])
B_i = dict([(i, 0) for i in item_list]) 
# lambda = 1
l = 1
old_error_reg = 0
new_error_reg = calc_error_reg(a, B_u, B_i, l, data)
#"""
iter = 0;
while math.fabs(old_error_reg - new_error_reg) > 0.0001 * old_error_reg:
    a = numpy.mean([x['rating'] - (B_u[x['reviewerID']] + B_i[x['itemID']]) for x in data]) 
    for u in user_list:
        B_u[u] = sum([i['rating'] - (a + B_i[i['ID']]) for i in I_u[u]]) / (l + len(I_u[u]))
    for i in item_list:
        B_i[i] = sum([u['rating'] - (a + B_u[u['ID']]) for u in U_i[i]]) / (l + len(U_i[i]))
    old_error_reg = new_error_reg
    new_error_reg = calc_error_reg(a, B_u, B_i, l, data)
    # print "new_error_reg: " + str(new_error_reg)
    iter += 1
# print "Iterations: " + str(iter)
# Item bias B_I102776733
t2_1 = B_i['I102776733']
print "\tItem bias of I102776733 = " + str(t2_1)
# User bias B_U566105319
t2_2 = B_u['U566105319']
print "\tUser bias of U566105319 = " + str(t2_2)
# MSE of the predictor against the test data ('labeled_Rating.txt')
test_rating = [line.strip().split(' ') for line in open("labeled_Rating.txt")]
mse = numpy.mean([(float(y[2]) - (a + B_u[y[0]] + B_i[y[1]])) ** 2 for y in test_rating if y[0] in user_list and y[1] in item_list])
t2_3 = mse
print "\tMSE of predictor against the test data: " + str(t2_3)
############################################################################
# NUM 2 FAIL
# X = [(l['reviewerID'],l['rating']) for l in data]
# Y = [(l['itemID'],l['rating']) for l in data]
# res = defaultdict(list)
# res2 = defaultdict(list)
# for v, k in X: res[v].append(k)
# for a, b in Y: res2[a].append(b)

# #rez has all the means
# rez = res
# rez2 = res2
# for u in res:
# 	rez[u] = (sum(res[u])/len(res[u])) - ymean
# for i in res2:
# 	rez2[i] = (sum(res2[i])/len(res2[i])) - ymean

# #ANSWER 2	
# ans2 = ymean + rez2['I102776733'] + rez['U566105319']
# print "\n<><><>Problem 2<><><>"
# print "\tItem bias of I102776733 = " + str(rez2['I102776733'])
# print "\tUser bias of U566105319 = " + str(rez['U566105319'])
# #MSE
# #regular
# # for p in rez:
# MSK = 0
# for f in full:
# 	MSK += ((f[2] - (ymean + rez[f[0]] + rez2[f[1]]))**2)/len(f)
# #ANSWER
# print "\tMSE of predictor against the test data: " + str(MSK)

##################################Problem 3#################################
#U229891973
#U622491081
X = [(l['reviewerID'],l['itemID']) for l in data]
res = defaultdict(list)
for v, k in X: res[v].append(k)
res['U229891973']
res['U622491081']

#Find common elements
def jax(u1,u2):
	common = set(res[u1]) & set(res[u2])
	numCommon = len(common)
	union = set(res[u1]) | set(res[u2])
	numUnion = len(union)
	#JACCARD
	jacc = float(numCommon) / float(numUnion)
	return jacc
	
print jax('U229891973', 'U622491081')# ANSWER PART A
print "\n<><><>Problem 3<><><>"
print ("\tJaccard similarity between the users 'U229891973' and 'U622491081' is: "
		+ str(jax('U229891973', 'U622491081')))
high = 0
jak = []
for u in res:
	if u == 'U622491081':
		continue
	jac = jax(u, 'U622491081')
	if jac > high:
		high = jac
		jak = [u]
	elif jac == high:
		jak.append(u)
print ("\tUsers with the highest Jaccard similarity to 'U622491081' are: "
		+ str(jak))

# #TEST to see if corrrect
# for j in jak:
# 	print jax(j, 'U622491081')
		
##################################Problem 4#################################
#foreach -- sum(helpful) / len(helpful) = value <-- find the mean of that
avg =[]
for l in data:
	helpful,total = l['helpful']['nHelpful'],l['helpful']['outOf']
	avg.append(float(helpful)/float(total))
a = sum(avg)/len(avg)# ANSWER PART A

print "\n<><><>Problem 4<><><>"
print "\tFitted value of alpha is: " + str(a)

#MSE for part B
test_helpful = [line.strip().split(' ') for line in open("labeled_Helpful.txt")]
mse = numpy.mean([(float(y[3]) - (a * float(y[2]))) ** 2 for y in test_helpful])
t4_2 = mse 
print "\tMSE: " + str(t4_2)

#AE for part B
ae = sum([math.fabs(float(y[3]) - (a * float(y[2]))) for y in test_helpful])
t4_3 = ae
print "\tAbsolute error: " + str(t4_3)

# part C
def feature(datum):
    feat = [1]
    feat.append(len(datum['reviewText'].split(' ')))
    feat.append(datum['rating'])
    return feat
X = [feature(d) for d in data]
y = [float(d['helpful']['nHelpful']) / float(d['helpful']['outOf']) for d in data]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
t4_4 = theta
print "\tFitted parameters are ([a, Bias1, Bias2]): " + str(t4_4)

#PART D





# print "Reading \"helpful.json.gz\"..."
# test_helpful = open("helpful.json")
# print "Done."
# mse = numpy.mean([(float(x['helpful']['nHelpful']) - (theta[0] + theta[1] * len(y['reviewText'].split(' ')) + theta[2] * y['rating'])) ** 2 for x, y in zip(data, test_helpful)])
# t4_5 = mse 
# print "MSE: " + str(t4_5)
# ae = sum([math.fabs(float(x['helpful']['nHelpful']) - (theta[0] + theta[1] * len(y['reviewText'].split(' ')) + theta[2] * y['rating'])) for x, y in zip(data, test_helpful)])
# t4_6 = ae
# print "Absolute error: " + str(t4_6)