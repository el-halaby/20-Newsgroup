# 20-Newsgroup
Classification of text documents into 20 classes (topics) using only the subject of each document. The idea here is to avoid using the entire post, and just use the subject line to classify each document.

After keeping only the subject of each document, TfidfVectorizer was used to extract features.

The obtained features are then fed to MaxAbsScaler for feature scaling, then finally, feature selection using SelectFromModel (with a LinearSVC model) is performed.

Cross validation (10 folds) was used to evaluate the model. The mean score was Mean: 96.79% with a standard deviation of 0.31%. 
