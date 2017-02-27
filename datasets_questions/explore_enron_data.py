#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
poi_data = open("../final_project/poi_names.txt", "r").readlines();


totalCount = len(enron_data);
poiCount = 0;
salaryCount = 0;
emailAddressCount = 0;
totalPaymentsCount = 0;
poiAndNoTotalPaymentCount = 0;
for key in enron_data:
    #print key;
    person = enron_data[key];
    print enron_data[key];
    if person["poi"] == 1:
        poiCount = poiCount + 1;
    if type(person["salary"]) is int:
        salaryCount = salaryCount + 1;
    if person["email_address"] <> "NaN":
        emailAddressCount = emailAddressCount + 1;
    if person["total_payments"] <> "NaN":
        totalPaymentsCount = totalPaymentsCount + 1;
    if person["poi"] == 1 and person["total_payments"] == "NaN":
        poiAndTotalPaymentCount = poiAndNoTotalPaymentCount + 1;
        
    #print len(enron_data[key]);
print "Total Count: " + str(totalCount);    
print "Poi Count: " + str(poiCount);
print "Salary Count: " + str(salaryCount);
print "Email Address Count: " + str(emailAddressCount);
print "Total Payment Count: " + str(totalPaymentsCount);
print "Total Payment Ratio: " + str(float(totalPaymentsCount) / totalCount);
print "Total No Payment Ratio: " + str(1.0 - float(totalPaymentsCount) / totalCount)
print "Poi & No Total Payment Count: " + str(poiAndNoTotalPaymentCount);
print "Poi & No Total Payment Ratio: " + str(poiAndNoTotalPaymentCount / totalCount);

poiCountInNames = 0;
for line in poi_data:
    if (len(line) >= 3 and line[:1] == '(') :
        poiCountInNames = poiCountInNames + 1;
    #print line;
print poiCountInNames;

print enron_data["PRENTICE JAMES"]["total_stock_value"];
print enron_data["Colwell Wesley".upper()]["from_this_person_to_poi"];
print enron_data["Skilling Jeffrey K".upper()]["exercised_stock_options"];
