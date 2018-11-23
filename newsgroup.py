import os
from sklearn.model_selection import cross_val_score
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn.preprocessing import scale, StandardScaler, Normalizer, normalize, MinMaxScaler, MaxAbsScaler
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def by_subject(filename, cls):
    data = open(filename)
    lines = [line for line in data]
    X = []
    for i in range(0,len(lines)):
        if lines[i].startswith('Document_id:'):
            if len(lines[i+1].split('Subject:')) > 1:
                X.append(lines[i+1].split('Subject:')[1])
            else:
                X.append(lines[i+2].split('Subject:')[1])
    target = [cls for i in range(0, len(X))]
    return X, target

def by_email(filename, cls):
    data = open(filename)
    lines = [line for line in data]
    X = []
    for i in range(0,len(lines)):
        if lines[i].startswith('Document_id:'):
            if len(lines[i+1].split('From:')) > 1:
                X.append(lines[i+1].split('From:')[1])
            else:
                X.append(lines[i+2].split('From:')[1])
    target = [cls for i in range(0, len(X))]
    return X, target 


def by_both(filename, cls):
    data = open(filename)
    lines = [line for line in data]
    X = []
    print('length:',len(lines))
    for i in range(0,len(lines)):
        if lines[i].startswith('Document_id:'):
            if len(lines[i+1].split('From:')) > 1 and len(lines[i+2].split('Subject:')) > 1:
                X.append(lines[i+1].split('From:')[1] + ' ' + lines[i+2].split('Subject:')[1])
            else:
                X.append(lines[i+1].split('Subject:')[1] + ' ' + lines[i+2].split('From:')[1])
    target = [cls for i in range(0, len(X))]
    return X, target

if __name__ == '__main__':
    # loop over all text files
    directory = './Data/'
    X = []
    Y = []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):    
            print('Reading:', filename)
            #data, target = by_email(directory+filename, count) 
            #data, target = by_subject(directory+filename, count)
            data, target = by_both(directory+filename, count)
            for x, y in zip(data, target):
                X.append(x)
                Y.append(y)
            count = count + 1

    # shuffle
    z = list(zip(X,Y))
    random.shuffle(z)
    X,Y = zip(*z)
    print('Classes:', set(Y))

    print('Number of documents:', len(X), '\n')
    # feature extraction
    X_new = TfidfVectorizer(stop_words = 'english',  sublinear_tf=True).fit_transform(X)

    # feature transformation
    #X_new = Normalizer().fit_transform(X_new)
    X_new = MaxAbsScaler().fit_transform(X_new)
    print(X_new.shape)
    
    # feature selection
    X_new = SelectFromModel(LinearSVC(penalty='l1', dual=False), threshold='0.4*mean').fit_transform(X_new, Y)
    print(X_new.shape)

    clf = LinearSVC()
    results = cross_val_score(clf, X_new, Y, cv=10, n_jobs = 10)
    print('Mean:', results.mean()*100)
    print('STD:', results.std()*100)
