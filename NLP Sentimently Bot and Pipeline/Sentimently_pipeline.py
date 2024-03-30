import pandas as pd
import re
from vnlp import Normalizer
from vnlp import StopwordRemover
from vnlp import StemmerAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def word_stemmer(text,stemmer):
    """
    finds stem of word

    Parameters
    ----------
    text : list
        list of word
    stemmer : vlnp StemmerAnalyzer
        object of vnlp stemmer analyzer

    Returns
    -------
    list
        list of edited word

    """
    a=stemmer.predict(text)
    liste=[]
    for i in range(0,len(a)):
        b=a[i]
        liste.append(b[0:b.find("+")])
    return " ".join(i for i in liste)

def data_preparation(df,normalizer,stemmer,stopword_remover,tf_idf_word_vectorizer,rare_words=1,only_text_preprocessing=False):
    """
    prepares tha data for the model 
    
    Parameters
    ----------
    df : Data Frame
        data to be processed
    normalizer : vnlp Normalizer
        object of vnlp nornalizer
    stemmer : vlnp StemmerAnalyzer
        object of vnlp stemmer analyzer
    stopword_remover : vnlp StopwordRemover
        object of vlnp stop word remover
    tf_idf_word_vectorizer : TfidfVectorizer
        object of sklearn.feature_extraction.text TfidfVectorizer
    rare_words : integer
        amount of rare words to remove. The default is one.
    only_text_preprocessing : True or False
        to do only text preprocessing. The default is False.

    Returns
    -------
    if only_text_preprocessing=False
        X_train, X_test
            TF-IDF matrix
        y_train, y_test
            dependent variable
        tf_idf_word_vectorizer
            object of tf_idf_word_vectorizer        
    if only_text_preprocessing=True
        df
            text preprocessed data only

    """
    df["tweet"]=df["tweet"].apply(lambda x: Normalizer.lower_case(str(x)))
    df["tweet"]=df["tweet"].apply(lambda x: Normalizer.remove_accent_marks(str(x))) 
    df["tweet"]=df["tweet"].apply(lambda x: re.sub("[^a-zıöüşçğ]"," ",str(x)))
    df["tweet"]=df["tweet"].apply(lambda x: " ".join(Normalizer.deasciify(str(x).split()))) 
    df["tweet"]=df["tweet"].apply(lambda x: " ".join(normalizer.correct_typos(str(x).split()))) 
    df["tweet"]=df["tweet"].apply(lambda x: word_stemmer(x,stemmer)) 
    df["tweet"]=df["tweet"].apply(lambda x: " ".join(stopword_remover.drop_stop_words(str(x).split())))
    
    temp_df=pd.Series(" ".join(df["tweet"]).split()).value_counts()
    drops=temp_df[temp_df<=rare_words]
    df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))

    if only_text_preprocessing==False:
        y=df["label"]
        X=df["tweet"]
            
        x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=14)
            
        tf_idf_word_vectorizer=tf_idf_word_vectorizer.fit(x_train)
        X_train=tf_idf_word_vectorizer.transform(x_train)
        X_test=tf_idf_word_vectorizer.transform(x_test)
        
        return X_train,X_test,y_train,y_test,tf_idf_word_vectorizer
    else:
        return df

def logistic_regression(X_train, X_test, y_train, y_test):
    """
    establishing the model of logistic regression

    Parameters
    ----------
    X_train : Tf-idf Vectorizer Matrix
        Tf-idf Vectorizer Matrix for train
    y_train : integer
        dependent variable

    Returns
    -------
    log_model : object of the model

    """
    log_model=LogisticRegression()
    log_model.fit(X_train,y_train)
    print("Logistic Regession CV Accuracy Mean: ", cross_val_score(log_model,
                                                                   X_train,
                                                                   y_train,
                                                                   scoring="accuracy",
                                                                   cv=5).mean())
    y_pred=log_model.predict(X_test)
    print(classification_report(y_pred,y_test))
    return log_model

def predict_new_tweet(dataframe_new, log_model, tf_idf_word_vectorizer):
    """
    Predicting the sentiment (positive, negative or neutral) of new edited tweets 
    with the established Logistic Regression model

    Parameters
    ----------
    dataframe_new : dataframe containing tweets arranged to enter the model
        
    log_model : object of Logistic Regression
        
    tf_idf_word_vectorizer : object of tf_idf_word_vectorizer
        

    Returns
    -------
    dataframe_new : Dataframe containing username, tweet and labels predicted by the model

    """
    tweet_tfidf = tf_idf_word_vectorizer.transform(dataframe_new["tweet"])
    predictions = log_model.predict(tweet_tfidf)
    dataframe_new["label"] = predictions
    return dataframe_new

def main():
    dataframe = pd.read_csv("C:\\Users\\Dell\Desktop\\NLP\\sentimenty bot\\tweets_labeled.csv")
    normalizer = Normalizer()
    stemmer = StemmerAnalyzer()
    stopword_remover = StopwordRemover()
    tf_idf_word_vectorizer=TfidfVectorizer()
    X_train, X_test, y_train, y_test, tf_idf_word_vectorizer = data_preparation(dataframe,
                                                                                normalizer,
                                                                                stemmer,
                                                                                stopword_remover,
                                                                                tf_idf_word_vectorizer)
    log_model = logistic_regression(X_train, X_test, y_train, y_test)
    dataframe_new = pd.read_csv("C:\\Users\\Dell\\Desktop\\NLP\\sentimenty bot\\tweets_21.csv")    
    dataframe_new=data_preparation(dataframe_new, normalizer,stemmer,stopword_remover,tf_idf_word_vectorizer,
                                   only_text_preprocessing=True)                     
                                                                                                               
    predicted_df = predict_new_tweet(dataframe_new, log_model, tf_idf_word_vectorizer)
    print(predicted_df)
    

if __name__ == "__main__":
    print("The process has started.")
    main()

