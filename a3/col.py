# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 00:18:34 2016

@author: roger
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:25:31 2016

@author: vaishnavithangapalam
"""
import csv
import os
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import itertools
import operator
from TwitterAPI import TwitterAPI
import json
import pandas as pd

consumer_key = '3GAqwEymKlq8mQnlfSXcd7h9g'
consumer_secret = 'H5OLTvPmPXl5kqL7hA4XpuwQplbTriRcnORvPzZgtZXRxK1Ip4'
access_token = '738905027574693888-bblQ0GW8iloGTKTN6W9LDrGwXd0y8pr'
access_token_secret = 'HDiODGNdPkNcxqEqycyWrUdui7Qi0gC3Ox9IVJJny0UlX'


# This method is done for you. Make sure to put your credentials in thepip install python_twitter file twitter.cfg.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you enteraed above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

    
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)
    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup
    In this example, I test retrieving two users: twitterapi and twitter.
    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO
    alldata =[]
    resource = 'statuses/user_timeline'
    tweets=[]
    n_tweets=1000
    params = {'screen_name': 'realDonaldTrump','count':200}
    """trumpTweetsResponse = robust_request(twitter, resource, params, max_tries=2)
    for r in trumpTweetsResponse:
        if(not(r['text'].startswith( 'RT' ))):
            arr=[]
            arr.append(r['id_str'])
            arr.append(r['text'])
            arr.append(0)
            lastid =r['id_str']
            alldata.append(arr)
             
    val = len(alldata)
    
    
    while(val <n_tweets):
        #params =  {'q': '@realDonaldTrump','count':100,'max_id':int(lastid)-1}
        params = {'screen_name': 'realDonaldTrump','count':200,'max_id':int(lastid)}
        res=robust_request(twitter, resource, params, max_tries=2)
        for r in res :
                                  
            if(not(r['text'].startswith( 'RT' ))):
                arr=[]
                arr.append(r['id_str'])
                arr.append(r['text'])
                arr.append(0)
                lastid =r['id_str']
                alldata.append(arr)
       
        val = len(alldata)
          
        if len(tweets) >= n_tweets:
            break
        
    print('fetched %d tweets' % len(alldata))
         

    print(len(alldata))
    with open('traindata_fin.csv', 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(('ID','DATA','SENTIMENT'))  # only if you want it
        for t in alldata:
            writer.writerow(t)"""    
    
    #file = open('traindata_fin.csv', 'r')
    #print(file.read())
   
    dataTest = pd.read_csv('traindata_fin.csv')
    print(dataTest)
    return alldata
    
    
    
    
def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()

    get_users(twitter)
 

if __name__ == '__main__':
    main()