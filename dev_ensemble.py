import pandas as pd
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import dev_sugeno_integral

def getfile(filename, root="../"):
    file = root+filename+'.csv'
    df = pd.read_csv(file,header=None)
    df = np.asarray(df)

    labels=[]
    for i in range(115): #covid 376, metal abnormal 45
        labels.append(0)
    for i in range(115): #non-COVID 369, metal normal 22
        labels.append(1)
    labels = np.asarray(labels)
    return df,labels

def get_Lap_file(filename, root="../"):
    file = root+filename+'.csv'
    df = pd.read_csv(file,header=None)
    df = np.asarray(df)

    return df


def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction

def metrics(labels,predictions,classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes,digits = 4))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("\nBalanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))

#Sugeno Integral
def ensemble_sugeno(labels,prob1,prob2,prob3,prob4, prob1_Lap, prob2_Lap, prob3_Lap, prob4_Lap):
    num_classes = prob1.shape[1]
    Y = np.zeros(prob1.shape,dtype=float)
    for samples in range(prob1.shape[0]):
        for classes in range(prob1.shape[1]):
            X = np.array([prob1[samples][classes], prob2[samples][classes], prob3[samples][classes], prob4[samples][classes] ])
            RATIO = np.array([prob1_Lap[samples][0], prob2_Lap[samples][0], prob3_Lap[samples][0], prob4_Lap[samples][0]])

            # measure = np.array([1.5, 1.5, 0.01, 1.2])

            #밝기와 어둡기는 제외하고 블러와 정상 데이터로만 실험 진행
            # X = np.array([prob1[samples][classes]])
            # RATIO = np.array([prob1_Lap[samples][0]])

            # X_agg = dev_sugeno_integral.sugeno_fuzzy_integral_generalized(X,measure)

            # X_agg = dev_sugeno_integral.calculate_mean(X)
            # X_agg = dev_sugeno_integral.calculate_max(X)

            X_agg = dev_sugeno_integral.calculate_proposedMethod(X, RATIO)
            # X_agg = dev_sugeno_integral.calculate_max_proposedMethod(X, RATIO)
            Y[samples][classes] = X_agg

    sugeno_pred = predicting(Y)

    correct = np.where(sugeno_pred == labels)[0].shape[0]
    total = labels.shape[0]

    print("Ensemble Accuracy = ",correct/total)
    classes = ['Aboraml','Normal']
    metrics(sugeno_pred,labels,classes)
