# Text document classification of "20 Newsgroups" dataset using Metadata
Classification of text documents from the "20-newsgroups" dataset using only the subject or sender's email (or both) of each document. The idea here is to avoid using the entire post, and just use metadata from posts to classify each document, which hasn't been done to the best of my knowledge.

### Requirements
* Python 3.6.1 or above
* sklearn

### Model
The model used was LinearSVC with l2 penalty. 

### Feature Extraction
After keeping only the subject (or email) of each document, TfidfVectorizer was used to extract features. In both cases (classification by subject or email), the obtained features are then fed to MaxAbsScaler for feature scaling.

### Feature Selection
Feature selection using SelectFromModel (with a LinearSVC model) is performed.

### Model Evaluation
Cross validation (10 folds) was used to evaluate the model.
* Classification by sender's email: The mean score was 64.91% with a standard deviation of 1.26%.
* Classification by post's subject: The mean score was 88.25% with a standard deviation of 0.88%.
* Classification by both post's subject and sender's email: The mean score was 91.22% with a standard deviation of 0.55%.

### Directory Setup and Dataset
The "Data" directory contains 20 text files, each file contains a number of posts relating to a single subject. The dataset was downloaded from:
https://www.kaggle.com/crawford/20-newsgroups.
To run the program, make sure to download the "Data" folder and execute the script "newsgroup.py".

If you find any errors, mistakes or bugs, please let me know. Feel free to comment.
