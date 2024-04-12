

from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import time
import numpy as np 
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, accuracy_score, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.multiclass import unique_labels
import nltk
from collections import defaultdict
from datetime import datetime
import sys

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# Get English stopwords
stop_words = set(stopwords.words('english'))
recordsToRead=1000





# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()
print("noOfrecords=",recordsToRead)

# Function to remove stop words from text
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


# Function to lemmatize text
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]  # Lemmatize words
    return ' '.join(lemmatized_tokens)








# class NaiveBayes:
#     def __init__(self):
#         self.class_probabilities = {}
#         self.word_probabilities = defaultdict(dict)
#         self.classes = None
    
#     def fit(self, X, y):
#         # Calculate prior probabilities
#         self.classes, class_counts = np.unique(y, return_counts=True)
#         total_samples = len(y)
#         self.class_probabilities = {cls: count / total_samples for cls, count in zip(self.classes, class_counts)}
        
#         # Calculate conditional probabilities
#         vectorizer = CountVectorizer()
#         X_bow = vectorizer.fit_transform(X)
#         for cls in self.classes:
#             cls_indices = np.where(y == cls)[0]
#             cls_word_counts = X_bow[cls_indices].sum(axis=0)
#             total_words_in_class = cls_word_counts.sum()
#             for word, idx in vectorizer.vocabulary_.items():
#                 self.word_probabilities[word][cls] = (cls_word_counts[0, idx] + 1) / (total_words_in_class + len(vectorizer.vocabulary_))
    
#     def predict(self, X):
#         predictions = []
#         vectorizer = CountVectorizer(vocabulary=self.word_probabilities.keys())
#         X_bow = vectorizer.transform(X)
#         for i in range(X_bow.shape[0]):
#             scores = {cls: np.log(self.class_probabilities[cls]) for cls in self.classes}
#             for word, idx in vectorizer.vocabulary_.items():
#                 if X_bow[i, idx] > 0:  # Word is present in the document
#                     for cls in self.classes:
#                         scores[cls] += np.log(self.word_probabilities[word][cls])
#             predictions.append(max(scores, key=scores.get))
            
#             #print("self.class_probabilities:", self.class_probabilities)
#             #print("self.word_probabilities:", self.word_probabilities)
#         return predictions
    
class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.word_probabilities = defaultdict(dict)
        self.classes = None
    
    def fit(self, X, y):
        # Calculate prior probabilities
        self.classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        self.class_probabilities = {cls: count / total_samples for cls, count in zip(self.classes, class_counts)}
        
        # Calculate conditional probabilities
        vectorizer = CountVectorizer()
        X_bow = vectorizer.fit_transform(X)
        self.vocab_size = len(vectorizer.vocabulary_)
        for cls in self.classes:
            cls_indices = np.where(y == cls)[0]
            cls_word_counts = X_bow[cls_indices].sum(axis=0)
            total_words_in_class = cls_word_counts.sum()
            for word, idx in vectorizer.vocabulary_.items():
                self.word_probabilities[word][cls] = (cls_word_counts[0, idx] + 1) / (total_words_in_class + self.vocab_size)
    
    def predict(self, X):
        predictions = []
        vectorizer = CountVectorizer(vocabulary=self.word_probabilities.keys())
        X_bow = vectorizer.transform(X)
        for i in range(X_bow.shape[0]):
            scores = {cls: np.log(self.class_probabilities[cls]) for cls in self.classes}
            for word, idx in vectorizer.vocabulary_.items():
                if word in self.word_probabilities:  # Check if the word exists in vocabulary
                    if X_bow[i, idx] > 0:  # Word is present in the document
                        for cls in self.classes:
                            scores[cls] += np.log(self.word_probabilities[word][cls])
                    else:  # Word is not present in the document
                        for cls in self.classes:
                            scores[cls] += np.log(1 - self.word_probabilities[word][cls])
                else:  # Word is not present in the training data
                    for cls in self.classes:
                        scores[cls] += np.log(1 / (self.vocab_size + 1))
            predictions.append(max(scores, key=scores.get))
        return predictions, self.class_probabilities
    
    
    

def main(parameter):
    print("Parameter passed:", parameter)
    TRAIN_SIZE = float(parameter)
    if(TRAIN_SIZE < 0 or TRAIN_SIZE > 100):
        TRAIN_SIZE=80
    
    print("TRAIN_SIZE=", TRAIN_SIZE)
    testSize= 1 - (TRAIN_SIZE/100)
    toRunNBClassifier=True
    if toRunNBClassifier==True:
        
        # read data from the file
        
        file_review="yelp_academic_dataset_review.json"
        data = []
        count=0
        #data_file = open(dir + file_review, encoding="utf8")
        data_file = open(file_review, encoding="utf8")
        start_time = time.time()
        list_text = []
        list_stars = []
        for line in data_file:
            loadedLine=json.loads(line)
            list_text.append(loadedLine['text'])
            list_stars.append(loadedLine['stars'])
            count=count+1
            if(count==recordsToRead):
                break
        dfReview = pd.DataFrame({'text': list_text, 'stars': list_stars})
        data_file.close()

        # dfReview.stars.unique()
        # print("value counts for stars column:")
        # dfReview.stars.value_counts()
        print("The Stars/ratings distribution in dataset:")
        values, counts = np.unique(dfReview['stars'], return_counts=True)
        print("values=",values)
        print("counts=",counts)

        plt.figure()
        plt.bar(values, counts, tick_label=['1','2','3','4','5'])
        plt.title('Distribution of Stars')
        plt.xlabel('Stars')
        plt.ylabel('Number of reviews')
        plt.show()

        dfReview['sentiment'] = ''
        dfReview['sentiment'] = dfReview['stars'].apply(lambda x: 0 if (x >= 0 and x < 4) else 1)
        dfReview=dfReview.drop(columns=['stars'])
        dfReview.sentiment.unique()
        print("value counts for sentiment column:")
        print(dfReview.sentiment.value_counts())

        # Apply remove_stopwords function to the 'text' column
        dfReview['text'] = dfReview['text'].apply(remove_stopwords)

        #change all strings to be lower
        dfReview['text']=dfReview['text'].str.lower()

        #get rid of unwanted characters such as punctuation marks
        dfReview['text']=dfReview['text'].str.replace('[^\w\s]','')

        #removing digits
        dfReview['text']=dfReview['text'].str.replace('\d+','') 

        #removing any non-English characters
        dfReview['text']=dfReview['text'].str.replace('[^a-zA-Z]',' ')

        #removing line breaks
        dfReview['text']=dfReview['text'].str.replace('\n',' ').str.replace('\r','')

        #removing non-ascii characters
        dfReview['text']=dfReview['text'].str.replace('[^\x00-\x7f]','')

        #removing hyper-links
        dfReview['text']=dfReview['text'].str.replace('https?:\/\/.*[\r\n]*','')

        #removing dates
        dfReview['text']=dfReview['text'].str.replace('\d+[\.\/-]\d+[\.\/-]\d+','')
        dfReview['text']=dfReview['text'].str.replace('\d+[\.\/-]\d+[\.\/-]\d+','')
        dfReview['text']=dfReview['text'].str.replace('\d+[\.\/-]\d+[\.\/-]\d+','')
        
        print("before lemmatization:",datetime.now())
        # Apply lemmatize_text function to the 'text' column
        dfReview['text'] = dfReview['text'].apply(lemmatize_text)
        print("after lemmatization:",datetime.now())
        
        #split df
        x_trainDf, x_testDf, y_train, y_test = train_test_split(dfReview['text'], dfReview['sentiment'], test_size=testSize, random_state=42, shuffle=False, stratify = None)
        
        print("The sentiment distribution in training dataset:")
        y_trainDf = pd.DataFrame({'sentiment': y_train})
        values, counts = np.unique(y_trainDf['sentiment'], return_counts=True)
        print("values=",values)
        print("counts=",counts)
        plt.figure()
        plt.bar(values, counts, tick_label=['0','1'])
        plt.title('Distribution of sentiment in training dataset')
        plt.xlabel('sentiment')
        plt.ylabel('Number of reviews')
        plt.show()
        
        print("The sentiment distribution in test dataset:")
        y_testDf = pd.DataFrame({'sentiment': y_test})
        values, counts = np.unique(y_testDf['sentiment'], return_counts=True)
        print("values=",values)
        print("counts=",counts)
        plt.figure()
        plt.bar(values, counts, tick_label=['0','1'])
        plt.title('Distribution of sentiment in test dataset')
        plt.xlabel('sentiment')
        plt.ylabel('Number of reviews')
        plt.show()


        X_trainList = x_trainDf.tolist()  
        y_trainList = y_train.tolist()  
        X_testList =x_testDf.tolist()  

        # Initialize and train the Naive Bayes classifier
        nb_classifier = NaiveBayes()
        
        print("before nb_classifier.fit:",datetime.now())
        nb_classifier.fit(x_trainDf, y_train)
        print("after nb_classifier.fit:",datetime.now())
        
        nb_classifier11=nb_classifier
        
        # Make predictions
        print("before nb_classifier.predict:",datetime.now())
        y_predList,class_probabilities = nb_classifier.predict(X_testList)
        print("after nb_classifier.predict:",datetime.now())
        
        # vectorizer = CountVectorizer()
        # vectorized_data = vectorizer.fit_transform(x_trainDf)
        # nb_classifier = MultinomialNB(alpha=0.1)
        # nb_classifier.fit(vectorized_data, y_train)
        # #evaluate model
        # y_predList = nb_classifier.predict(vectorizer.transform(x_testDf))
        
        


        yTrain_pred_df=pd.DataFrame({'y_test': y_test.tolist(), 'y_pred': y_predList})

        y_test = yTrain_pred_df['y_test']
        y_pred = yTrain_pred_df['y_pred']

        # Calculate True Positives (TP)
        TP = ((y_test == 1) & (y_pred == 1)).sum()

        # Calculate True Negatives (TN)
        TN = ((y_test == 0) & (y_pred == 0)).sum()

        # Calculate False Positives (FP)
        FP = ((y_test == 0) & (y_pred == 1)).sum()

        # Calculate False Negatives (FN)
        FN = ((y_test == 1) & (y_pred == 0)).sum()

        # Calculate Sensitivity (Recall)
        sensitivity = TP / (TP + FN)

        # Calculate Specificity
        specificity = TN / (TN + FP)

        # Calculate Precision
        precision = TP / (TP + FP)

        # Calculate Negative Predictive Value
        negativePredictiveValue = TN / (TN + FN)

        # Calculate Accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        # Calculate F-score (Harmonic mean of precision and recall)
        F_score = 2 * ((precision * sensitivity) / (precision + sensitivity))

        # Print the calculated metrics
        
        
        print("True Positives (TP):", TP)
        print("True Negatives (TN):", TN)
        print("False Positives (FP):", FP)
        print("False Negatives (FN):", FN)
        print("Sensitivity (Recall):", sensitivity)
        print("Specificity:", specificity)
        print("Precision:", precision)
        print("Negative Predictive Value:", negativePredictiveValue)
        print("Accuracy:", accuracy)
        print("F-score:", F_score)
        
        # Define the true labels (y_true) and the predicted labels (y_pred)
        y_true = yTrain_pred_df['y_test']
        y_pred = yTrain_pred_df['y_pred']

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
        
        
        
        
    toRunNBClassifier=False
    end_time = time.time()
    execution_time = end_time - start_time
    print("after execution, execution time=",execution_time)
    
    print("\n")
    # Ask the user to input a single sentence
    sentence = input("Please enter a single sentence: ")
    # Print the inputted sentence
    print("You entered:", sentence)    
    y_predicted,class_probabilities = nb_classifier.predict([sentence])
    print("y_predicted=",y_predicted)
    print("class_probabilities=",class_probabilities)
    
    print("\n")
    sentence2 = input("Do you want to add another sentence? ")
    # Print the inputted sentence
    print("You entered:", sentence2)  
    # Ask the user to input a single sentence
    sentence2 = input("Please enter a single sentence: ")
    # Print the inputted sentence
    print("You entered:", sentence2)    
    y_predicted,class_probabilities = nb_classifier.predict([sentence2])
    print("y_predicted=",y_predicted)
    print("class_probabilities=",class_probabilities)
    
    
    
    
    
    
if __name__ == "__main__":
    print("the program is running")
    if len(sys.argv) != 2:
        TRAIN_SIZE="80"
        print("Provided argument count is not equal to 1.Default Parameter passed=80")
        main(TRAIN_SIZE)
    else:
        print("log entry else block")
        parameter_value = sys.argv[1]
        main(parameter_value)




           
#@@@@@===========================


