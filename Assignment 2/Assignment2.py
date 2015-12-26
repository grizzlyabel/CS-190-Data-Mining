import gzip
import operator
import numpy
from collections import defaultdict
import urllib
import time
start_time = time.time()

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

def feature(datum):
  feat = [1]
  return feat

true = True
false = False

print "Reading data..."
userD = list(parseData("yelp_academic_dataset_user.json"))
businessD = list(parseData("yelp_academic_dataset_business.json"))
reviewD = list(parseData("yelp_academic_dataset_review.json"))
checkinD = list(parseData("yelp_academic_dataset_checkin.json"))
print "done"

#############################################################################

# Get all the business IDs
# businessIDs = []
# city_to_train = []

# for i in businessD:
# 	businessIDs.append([i['business_id'], i['city'], i['neighborhoods'], i['review_count'], i['stars']])
	# Get the training portion. More specifically get the businesses from these cities
	# if i['city'] == 'Phoenix' or i['city'] == 'Karlsruhe' or i['city'] == 'Edinburgh' or i['city'] == 'Montreal':
	# 	city_to_train.append([i['business_id'], i['city']])

# for i in businessD:
# 	businessLocation.append([i['city'], i['neighborhoods']])
"""
# Get number of checkins for each business
checkinDict = {}

for i in checkinD:
	checkinDict[i['business_id']] = i['checkin_info']
"""

# list of businesses with checkins
checkinBiz = []
# list of dictionaries of checkins per business
checkinDict = []
# sum of all checkins per business
checkinSum = []
# Used for the first 5000 of most popular businesses
bizTownSorted = []


# Add checkins to the list
for i in checkinD:
	checkinDict.append(i['checkin_info'])

# Add businesses to the list
for i in checkinD:
	checkinBiz.append(i['business_id'])

# Go through each dictionary in the list
for x in range(0, len(checkinDict)):
	# Sum values in each dictionary
	checkinSum.append(sum(checkinDict[x].values()))

# Merge businesses and number of checkins to dictionary
dicBizSums = dict(zip(checkinBiz, checkinSum))

# Used to sort by the value of the key
sortedCheckins = sorted(dicBizSums.items(), key=operator.itemgetter(1))

# Reverse to get the most popular at the beginning of the dictionary
sortedCheckins.reverse()
##############################################################################

# Get the businesses in only these cities: Edinburgh, Karlsruhe, Montreal, Phoenix
businessIDs = []
businessCities = []
ratings = []
reviews = []

for i in businessD:
	if i['city'] == 'Edinburgh' or i['city'] == 'Karlsruhe' or i['city'] == 'Montreal' or i['city'] == 'Phoenix':
		businessIDs.append(i['business_id'])
		businessCities.append(i['city'])
		ratings.append(i['stars'])
		reviews.append(i['review_count'])

# Dictionary of Business IDs and the four cities
BizIdCity = dict(zip(zip(businessIDs, businessCities, ratings),reviews))

# Dictionary with Business IDs as keys and list of [city, # checkins] as vals
# However, we have ALL Business IDs present, so need to get rid of values with
# either only [city] or [# checkins]
d = defaultdict(list)
for a, b in BizIdCity.items() + dicBizSums.items():
	d[a].append(b)

# Dictionary with [city, # checkin] pairs as values
dd = dict((key, value) for key, value in d.iteritems() if type(value) is not tuple or len(value) == 3)
#or len(value) == 3

X = [feature(d) for d in dd]
y = [d[''] for d in dd]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)


# To time the execution time
print("\n--- %s seconds ---" % (time.time() - start_time))