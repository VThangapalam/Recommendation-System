# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    indx = len(movies.columns)
    movies['tokens'] = np.empty((len(movies), 0)).tolist()
    for index,row in movies.iterrows():
        temp_token=tokenize_string(row['genres'])
        movies.set_value(index,'tokens',temp_token)
    return movies



def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    N = len(movies.index)
    vocab = defaultdict(lambda: None) 
    rowtf = defaultdict(lambda: defaultdict(lambda: 0))
    rowMaxtf = defaultdict(lambda:0)
    docTermFreq= defaultdict(lambda:0)
    vocabSet=set()
    vocabList=[]
    for index,row in movies.iterrows():
        for val in row['tokens']:
           vocabSet.add(val)
           #first occurence of word in doc
           if(rowtf[index][val]==0):
               docTermFreq[val]=docTermFreq[val]+1
           #update term freq    
           rowtf[index][val]=rowtf[index][val]+1
           if(rowMaxtf[index]<rowtf[index][val]):
               rowMaxtf[index]=rowtf[index][val]
        
    vocabList=sorted(list(vocabSet))    
    vocabLen = len(vocabList)
    #create the vocab
    for index,word in enumerate(vocabList):
        vocab[word]=index        
    movies['features'] =  csr_matrix( (1,vocabLen) ) 
    #create csr matrix for the features column    
    for index,row in movies.iterrows():
        indptr = []
        indices = []
        data = []
        for term,tf in rowtf[index].items(): 
            terFreq= tf/rowMaxtf[index]
            idf = math.log10(N/docTermFreq[term])
            tfidf = terFreq*idf
            indices.append(vocab[term])
            data.append(tfidf)
            indptr.append(0)
        arr=csr_matrix( (data,(indptr,indices)), shape=(1,vocabLen) )       
        movies.set_value(index,'features',arr)
    return (movies,vocab)


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    dotProd =  a.dot(b.transpose())
    eucA= 0
    eucB= 0
    sumSqrA=0
    sumSqrB=0
    #calulating the euclidean dist
    for val in a.data:
       sumSqrA=sumSqrA+math.pow(val, 2)
    eucA= math.sqrt(sumSqrA) 
    for val in b.data:
       sumSqrB=sumSqrB+math.pow(val, 2)
    eucB= math.sqrt(sumSqrB)
    sim = dotProd/(eucA*eucB)
    return sim


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    predArr =[]
    for index,row in ratings_test.iterrows():
        #get other ratings by the user        
        otherRatingByuser=ratings_train[ratings_train.userId==row.userId]
        sim=[]
        #get the csr matrix of the test movie
        csra=movies[movies.movieId==row.movieId].iloc[0]['features']  
        #calculate the similarity of the test movie with the other movies rated by user
        for ind,r in otherRatingByuser.iterrows():        
            csrb=movies[movies.movieId==r.movieId].iloc[0]['features']   
            corr=cosine_sim(csra,csrb).data
            mov = (r.movieId)
            li=corr.tolist()
            if(len(li)==0):
                li.append(0)
            li.append(mov)
            tupl=tuple(li)
            sim.append(tupl)
        #get the movies with positive correlation with the test movie
        sortedCorr = sorted(sim, key=lambda tup: -tup[0])
        top_movie_ids = [x[1] for x in sortedCorr]
        top_movie_corrs = [x[0] for x in sortedCorr] 
        top_movie_rating=[]
        #get the rating for the movies with positive correlation    
        for idval in top_movie_ids:
            top_movie_rating.append(otherRatingByuser[otherRatingByuser.movieId==idval].iloc[0]['rating'] ) 
        rat= np.array(top_movie_rating)
        corrs=np.array(top_movie_corrs)
        pred=0
        #take the weighted average of the movies with positive correlation
        if(sum(top_movie_corrs)>0):
            pred=np.dot(rat,corrs)/sum(top_movie_corrs)
            predArr.append(pred)
        #when there are no movies with pos correlation cal the avg rating of user          
        else:
            pred=sum(top_movie_rating)/len(top_movie_rating)
            predArr.append(pred)           
    prediction = np.array(predArr)    
    return prediction


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])
   
if __name__ == '__main__':
    main()
