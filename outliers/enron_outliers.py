#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL", 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below



data_dict_filtered = [];
for i, key in enumerate(data_dict):
    person = data_dict[key];
    if (person["salary"] <> "NaN" or person["bonus"] <> "NaN"):
        if (person["salary"] >= 1000000 and person["bonus"] >= 5000000):
            print (key, person);
            data_dict_filtered.append((key, person));

#employees = [];
#for i, key in enumerate(data_dict_filtered):
    #print key;
#    employees.append((key, data[i][0], data[i][1]));
#employees.sort(key=lambda item: item[1]);
#print employees[len(employees) - 1];

#print len(data);
#print len(data_dict_filtered)



for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()