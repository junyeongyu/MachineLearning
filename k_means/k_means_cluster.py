#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
#feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]#, feature_3]
data = featureFormat(data_dict, features_list )

def getMin(column):
    minValue = 999999999;
    for record in data:
        value = record[column]
        if value <> 'NaN' and value < minValue:
            minValue = value
    return minValue;
    
def getMax(column):    
    #minValue = 999999999;
    maxValue = -999999999;
    for record in data:
        #print key;
        #record = data[index];
        value = record[column]
        if value <> 'NaN':
            #if value < minValue:
            #    minValue = value
            if value > maxValue:
                maxValue = value
        
    #print minValue
    return maxValue

max_salary = getMax(1)
min_salary = getMin(1)
max_exercised_stock_options = getMax(2)
min_exercised_stock_options = getMin(2)

rescaled_salary = (200000. - min_salary) / (max_salary - min_salary)
rescaled_eso = (1000000. - min_exercised_stock_options) / (max_exercised_stock_options - min_exercised_stock_options)

print rescaled_salary
print rescaled_eso

#for record in data:
#    record[1] = record[1] / max_salary;
#    record[2] = record[2] / max_exercised_stock_options;

poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans;
from sklearn.preprocessing import MinMaxScaler
klf = KMeans(n_clusters=2);
scaler = MinMaxScaler()
scaled_finance_features = scaler.fit_transform(finance_features)
klf.fit(scaled_finance_features);
pred = klf.predict(scaled_finance_features)



### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"