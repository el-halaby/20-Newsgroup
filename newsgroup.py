from sklearn.model_selection import cross_val_score
import os 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn.preprocessing import scale, StandardScaler, Normalizer, normalize, MinMaxScaler, MaxAbsScaler
from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def by_subject(filename, cls):
    data = open(filename)
    lines = [line for line in data]
    print('len:',len(lines))
    X = []
    for i in range(0,len(lines)):
        if lines[i].startswith('Subject:') and lines[i-1].startswith('From:'):
            X.append(lines[i])
    target = [cls for i in range(0, len(X))]
    print('Count',filename,':', len(target))
    return X, target

def by_email(filename, cls):
    data = open(filename)
    lines = [line for line in data]
    print('len:',len(lines))
    X = []
    for i in range(0,len(lines)):
        if lines[i].startswith('From:') and lines[i+1].startswith('Subject:'):
            X.append(lines[i])
    target = [cls for i in range(0, len(X))]
    print('Count',filename,':', len(target))
    return X, target

if __name__ == '__main__':
    # loop over all text files
    directory = './Data/'
    X = []
    Y = []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):    
            print('filename:', filename)
            #data, target = by_email(directory+filename, count) 
            data, target = by_subject(directory+filename, count)
            for x, y in zip(data, target):
                X.append(x)
                Y.append(y)
            count = count + 1
    # shuffle
    z = list(zip(X,Y))
    random.shuffle(z)
    X,Y = zip(*z)

    # feature extraction
    X_new = TfidfVectorizer(stop_words = 'english').fit_transform(X)

    # feature transformation
    X_new = Normalizer().fit_transform(X_new)
    print(X_new.shape)
    
    # feature selection
    X_new = SelectFromModel(LinearSVC(penalty='l1', dual=False)).fit_transform(X_new, Y)
    print(X_new.shape)

    clf = LinearSVC(penalty='l2', dual=True)
    results = cross_val_score(clf, X_new, Y, cv=10, n_jobs = 10)
    print('Mean:', results.mean()*100)
    print('STD:', results.std()*100)
