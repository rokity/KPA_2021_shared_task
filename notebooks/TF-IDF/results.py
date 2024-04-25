import pandas as pd
from sklearn.metrics import accuracy_score,f1_score

results = pd.read_csv("results.csv")
# measure the accurancy of the model 
# accuracy = int(results[results["label"] == results["prediction"]].shape[0])
# print("accuracy")

print("accuracy : ",accuracy_score(results["label"], results["prediction"]))
print("f1_score : ",f1_score(results["label"], results["prediction"], average='macro'))
