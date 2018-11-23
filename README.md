# 20-Newsgroup
Classification of text documents into 20 classes (topics) using only the subject of each document. The idea here is to avoid using the entire post, and just use the subject line to classify each document, which hasn't been done to the best of my knowledge.

After keeping only the subject of each document, TfidfVectorizer was used to extract features.

The obtained features are then fed to MaxAbsScaler for feature scaling, then finally, feature selection using SelectFromModel (with a LinearSVC model) is performed.

Cross validation (10 folds) was used to evaluate the model. The mean score was Mean: 96.81% with a standard deviation of 0.31%.

The "Data" directory contains the 20 text files, each text file contains a number of posts. The dataset was downloaded from:
https://www.kaggle.com/crawford/20-newsgroups.
To run the program, make sure to download the "Data" folder and execute the script "newsgroup.py".
