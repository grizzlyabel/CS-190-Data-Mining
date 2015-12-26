import gzip
import numpy
import math
from collections import defaultdict
import urllib
import scipy.optimize
import random
import json

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

##########################TASK 2##########################
### Helpfulness baseline: similar to the above. Compute the global average helpfulness rate, and the average helpfulness rate for each user

allHelpful = []
userHelpful = defaultdict(list)
# totalRate = defaultdict(list)
count = 0
firstHelpful = []
lastHelpful = []
allRatings = []

for l in readGz("train.json.gz"):
  user,item,helpfulRate = l['reviewerID'],l['itemID'],l['helpful']
  allHelpful.append(l['helpful'])
  userHelpful[user].append(l['helpful'])

allHelpful.sort()

for x in allHelpful:
  count += 1
  if count > 890000:
    lastHelpful.append(x)
  else:
    firstHelpful.append(x)
  # if x['outOf'] in totalRate:
  #   totalRate[x['outOf']].append(x['nHelpful'])
  # else:
  #   if x['nHelpful'] != []:
  #     totalRate[x['outOf']] = [x['nHelpful']]

# Takes the average of the Helpful rates for all reviews in the train.
averageRate = sum([x['nHelpful'] for x in allHelpful]) * 1.0 / sum([x['outOf'] for x in allHelpful[430000:]])
firstAverage = sum([x['nHelpful'] for x in firstHelpful]) * 1.0 / sum([x['outOf'] for x in firstHelpful])
lastAverage = sum([x['nHelpful'] for x in lastHelpful]) * 1.0 / sum([x['outOf'] for x in lastHelpful])


# totalRate = defaultdict(list)
# for a in allHelpful:
#   if a['outOf'] in totalRate:
#     totalRate[a['outOf']].append(a['nHelpful'])
#   else:
#     if a['nHelpful'] != []:
# for a in totalRate:
#   totalRate[a] = (numpy.mean(totalRate[a])/a) - averageRate

predictions = open("predictions_Helpful.txt", 'w')
for l in open("pairs_Helpful.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i,outOf = l.strip().split('-')
  outOf = int(outOf)
  if u in userRate and outOf < 1:
    if outOf < 1:
      newRate = (outOf*userRate[u] + outOf*firstAverage*4)/5
      predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(newRate) + '\n')
    else:
      newRate = (outOf*userRate[u] + outOf*lastAverage*6)/7
      predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(newRate) + '\n')
  else:
    predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf*averageRate) + '\n')

predictions.close()

##############################################################################

#TASK #1
### Purchasing baseline: just rank which items are popular and which are not, and return '1' if an item is among the top-ranked
userCount = defaultdict(int)
itemCount = defaultdict(int)
totalPurchases = 0

allReview = []
allHelpful = []
userHelpful = defaultdict(list)
split_array = defaultdict(list)
I_u = {}
U_i = {}
return1 = set()
return2 = set()
return3 = set()

count = 0
active = 0


user_list = set()
item_list = set()

for l in list(readGz("train.json.gz")):
  user,item = l['reviewerID'],l['itemID']
  categories = l['category']
  #ratings = list['rating']
  allReview.append(l['reviewText'])
  allHelpful.append(l['helpful'])
  userHelpful[user].append(l['helpful'])
  user_list.add(user)
  item_list.add(item)


  itemCount[item] += 1
  userCount[user] += 1
  totalPurchases += 1
  if (user not in I_u): 
    I_u.update({user: [l]})
  else:
    I_u[user].append(l)
  if (item not in U_i):
    U_i.update({item: [l]})
  else:
    U_i[item].append(l)

mostPopular = [(itemCount[x], x) for x in itemCount]
mostPopular.sort()
mostPopular.reverse()

mostActive = [(userCount[x], x) for x in userCount]
mostActive.sort()
mostActive.reverse()

return4 = set()
count4 = 0

for ic, i in mostPopular:
  count4 += ic
  return4.add(i)
  if count4 > totalPurchases/3: break

return5 = set()
count5 = 0

# Modification 2 - consider how active a user is
threshold = totalPurchases/12 # 0.70136

for upc, u in mostActive:
  count5 += upc
  return5.add(u)
  if count > threshold: break


def category_words(category):
  return set([word for subcategory in category for word in subcategory])

def diff_gender_clothing(user, item):
  clothing = 'Clothing, Shoes & Jewelry'
  if user not in I_u or item not in U_i:
    return False 
  item_cw = category_words(U_i[item][0]['category'])
  if clothing in item_cw:
    for i in I_u[user]:
      i_cw = category_words(i['category'])
      if clothing not in i_cw:
        continue
      if ('Men' in i_cw and 'Women' in item_cw) or ('Women' in i_cw and 'Men' in item_cw):
        return True
  return False

# for i in return2:
#   for x in range(len(ratings)):
#     if ratings[x][2] >= 3.0:
#       return3.add(ratings[x][1])

# Find the most active users in the purchases
for uc, u in mostActive:
  active += uc
  return2.add(u)
  if active > totalPurchases/12: break

# Find the most popular items in the purchases
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > (totalPurchases*2)/3: break

# for x in allReview:
#   split_array.append(allReview.split(' '))
#   if 'amazing' in split_array or 'awesome' in split_array:
#     #return 1 for that user
#   elif 'horrible' in split_array or 'never again' in split_array:
#     #return 0 for that user


# PEARSON ALGO
def pearson_item(i1, i2):
  R_i1_avg = numpy.mean([u['rating'] for u in U_i[i1]])
  R_i2_avg = numpy.mean([u['rating'] for u in U_i[i2]])
  R_intersection = [{'i1': u['rating'], 'i2': v['rating']} for u in U_i[i1] for v in U_i[i2] if u['reviewerID'] == v['reviewerID']]
  numerator = (sum([(r['i1'] - R_i1_avg) * (r['i2'] - R_i2_avg) for r in R_intersection])) * 1.0
  denominator = (sum([(r['i1'] - R_i1_avg) ** 2 for r in R_intersection]) * sum([(r['i2'] - R_i2_avg) ** 2 for r in R_intersection])) ** 0.5
  return 0.0 if denominator == 0.0 else numerator / denominator 

def pearson(user, item):
  if user not in I_u or item not in U_i:
    return True 
  for i in I_u[user]:
    if math.fabs(pearson_item(i['itemID'], item)) > 0.5:
      return True
  return False

randomGuessCount = []

def makePrediction(user, item):
  if user not in user_list:
    # If we don't know about the user, then see how popular it is?
    randomGuessCount.append((user,item))
    return 1;
  if item not in item_list: 
    # If we don't know about the item, then see how active user is?
    randomGuessCount.append((user,item))
    return 1;
  otherItemsBoughtByUser = [entry['itemID'] for entry in I_u[user]]

  # Finally, If this item is similar enough to one of the other items this 
  # user bought, predict 1
  for otherItem in otherItemsBoughtByUser:
    similarity = pearson_item(i, otherItem)
    if math.fabs(similarity) > 0.5: 
      #print similarity
      return 1
  if item in return4 or user in return5:
    return 1
  return 0  

# To output results
predictions = open("predictions_Purchase.txt", 'w')
for l in open("pairs_Purchase.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  if i in return1 or u in return2 or pearson(u, i):
    if diff_gender_clothing(u, i) or makePrediction(u, i):
      predictions.write(u + '-' + i + ",0\n")
    else:
      predictions.write(u + '-' + i + ",1\n")
  else:
    predictions.write(u + '-' + i + ",0\n")

predictions.close()
