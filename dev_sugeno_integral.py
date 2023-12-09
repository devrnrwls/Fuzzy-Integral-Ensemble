import numpy as np

def generate_cardinality(N, p = 2):
    return [(x/ N)**p for x in np.arange(N, 0, -1)]


def sugeno_fuzzy_integral(X, measure=None, axis = 0, keepdims=True):
    if measure is None:
        measure = generate_cardinality(X.shape[axis])

    return sugeno_fuzzy_integral_generalized(X, measure, axis, np.minimum, np.amax, keepdims)


def sugeno_fuzzy_integral_generalized(X, measure, axis = 0, f1 = np.minimum, f2 = np.amax, keepdims=True):
    X_sorted = np.sort(X, axis = axis)
    return f2(f1(np.take(X_sorted, np.arange(0, X_sorted.shape[axis]), axis), measure), axis=axis, keepdims=keepdims)

def calculate_mean(X):
    mean = sum(X) / len(X)
    return mean

def calculate_max(X):
    result = max(X)
    return result

def calculate_proposedMethod(X, RATIO):

    ratio_list = []
    X_ratio_list = []

    for item in RATIO:
        ratio_list.append(item/sum(RATIO))

    # ratio_list = [0.8, 0.0, 0.1, 0.1]

    for index, value in enumerate(ratio_list):
        X_ratio_list.append(value * X[index])

    result = sum(X_ratio_list)
    return result

def calculate_max_proposedMethod(X, RATIO):

    ratio_list = []
    X_ratio_list = []

    for item in RATIO:
        ratio_list.append(item/sum(RATIO))

    # ratio_list = [0.8, 0.0, 0.1, 0.1]

    for index, value in enumerate(ratio_list):
        X_ratio_list.append(value * X[index])

    result = max(X_ratio_list)
    return result



def getfile(filename, root="../"):
    file = root+filename+'.csv'
    df = pd.read_csv(file,header=None)
    df = np.asarray(df)

    labels=[]
    for i in range(376): #covid
        labels.append(0)
    for i in range(369): #non-COVID
        labels.append(1)
    labels = np.asarray(labels)
    return df,labels