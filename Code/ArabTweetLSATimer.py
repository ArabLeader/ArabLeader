# Program to perform Latent Semantic Analysis on a corpus of tweets in Arabic
# Tom Rishel
# University of Southern Mississippi
# April 2018

import os, glob, sys, glob2
from gensim import corpora, models, similarities
from pathlib import Path
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# function to read the tweets from text files into a list of strings
# also generates an csv file assigning an index value to each document
def files2Docs(path):
    documents = []
    counter = 0
    o = open("documentIndex.csv", 'w')
    #for currentFile in glob2.glob('C:\\Users\\Tom Rishel\\Documents\\TKX\\twitter_screen_names_by_country\\**\\*.txt'):
    for currentFile in glob2.glob(str(path) + '\\**\\*.txt'):
        #print("current file is: " + currentFile)
        i = open(currentFile, 'r', encoding = "utf-8")
        documents.append(i.read())
        # testing output
        #if counter >= 120:
        #    if counter < 122:
        #        print (currentFile)
        #        print ("\n")
        #        print(str(documents[counter]).encode("utf-8"))
        #        print("\n\n")
        i.close()
        o.write(str(counter) + "," + str(currentFile) + "\n" )
        counter = counter + 1
    print("counter = " + str(counter))
    o.close()
    return documents


# function to read stop words from the supplied file and store them in a set
def getStopWords(file):
    i = open(file, 'r', encoding = "utf-8")
    stoplist = set(i.read().split())
    i.close()
    return stoplist

# parent function to call sub functions 
def ArabTweetBuildVectorCorpus(tweetFolder, stopFile):
    documents = files2Docs(tweetFolder)
    stoplist = getStopWords(stopFile)
    #remove stop words from documents
    texts = [[word for word in document.split() if word not in stoplist]
         for document in documents]
    dictionary = corpora.Dictionary(texts)
    dictionary.save('.\\Arabic_tweet_LSI\\arab_tweet.dict')  # store the dictionary, for future reference
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('.\\Arabic_tweet_LSI\\arab_tweet.mm', corpus)  # store to disk, for later use


# function to generate the LSI model - consumes considerable computer resources and time
def ArabTweetTransformCorpus():
    #print(str(os.getcwd()))
    if (os.path.exists(".\\Arabic_tweet_LSI\\arab_tweet.dict")):
        # load the dictionary from disk
        dictionary = corpora.Dictionary.load('.\\Arabic_tweet_LSI\\arab_tweet.dict')
        # load the corpus in Matrix Market format from disk
        corpus = corpora.MmCorpus('.\\Arabic_tweet_LSI\\arab_tweet.mm')
        # message to let us know that we successfully loaded the dictionary and corpus
        print("Used files generated from first tutorial")
        # create the tfidf-weighted space
        tfidf = models.TfidfModel(corpus)
        tfidf_corpus = tfidf[corpus]
        lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)
        lsi.save(".\\Arabic_tweet_LSI\\lsi.model")
        
    else:
        print("Please run first tutorial to generate data set")


#function to load the lsi model from disk and run similarities on hard-coded queries
# queryDocPath is the path and filename of the query document as a text file
# category is the category of the query (e.g. terror, soccer, religion, etc.)
def ArabTweetSimilarities(queryDocPath, category):
    # load the dictionary from disk
    dictionary = corpora.Dictionary.load('.\\Arabic_tweet_LSI\\arab_tweet.dict')
    # load the corpus in Matrix Market format from disk
    corpus = corpora.MmCorpus('.\\Arabic_tweet_LSI\\arab_tweet.mm')
    # create the tfidf-weighted space
    tfidf = models.TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]
    lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)
    #index = similarities.Similarity(lsi[corpus]) # transform corpus to LSI space and index it
    index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it
    index.save('.\\Arabic_tweet_LSI\\arab_tweet.index')

    # testing semantic vector format for possible import into a neural network
    test = lsi.get_topics()

    # open and read the file storing the query document
    i = open(queryDocPath, 'r', encoding = "utf-8")
    vec_bow = dictionary.doc2bow(i.read().split())
    vec_lsi = lsi[vec_bow] # convert the query to LSI space
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    o = open(category + "DocSims.csv", 'w')
    categoryList = []
    for sim in sims:
        docId, docSim = sim
        o.write(str(docId) + "," + str(docSim) + "\n")
    o.close()


# function to load the previously saved index, read the query document, calculate 
# the similarities, and save them to a csv file

# queryDocPath is the path and filename of the query document as a text file
# category is the category of the query (e.g. terror, soccer, religion, etc.)
def runQuery(queryDocPath, category):
    # load the dictionary from disk
    dictionary = corpora.Dictionary.load('.\\Arabic_tweet_LSI\\arab_tweet.dict')
    # load the corpus in Matrix Market format from disk
    corpus = corpora.MmCorpus('.\\Arabic_tweet_LSI\\arab_tweet.mm')
    # create the tfidf-weighted space
    tfidf = models.TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]
    lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)
    # load the previously saved index
    index = similarities.MatrixSimilarity.load('.\\Arabic_tweet_LSI\\arab_tweet.index')

    # testing semantic vector format for possible import into a neural network
    test = lsi.get_topics()

    # open and read the file storing the query document
    i = open(queryDocPath, 'r', encoding = "utf-8")
    vec_bow = dictionary.doc2bow(i.read().split())
    vec_lsi = lsi[vec_bow] # convert the query to LSI space
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    o = open(category + "DocSims.csv", 'w')
    categoryList = []
    for sim in sims:
        docId, docSim = sim
        o.write(str(docId) + "," + str(docSim) + "\n")
    o.close()
    

# run ArabTweetLSA <parent folder with Arabic tweets> <stop word file name> from the command line
if __name__ == '__main__':
    # to run similarities to a query using previously generated vector space use runQuery(<path to the query text document> <category of the query>)
    runQuery("C:\\Users\\Tom Rishel\\OneDrive - The University of Southern Mississippi\\Documents\\Research\\TKX\\query_terrorism.txt", "terror")

    # if running BuildVectorCorpus, pass the name of the parent directory containing 
    # the corpus data and the name of the stop word file
    # syntax is ArabTweetLSA <parent folder with Arabic tweets> <stop word file name>
    #ArabTweetBuildVectorCorpus(sys.argv[1], sys.argv[2])

    #ArabTweetBuildVectorCorpus("C:\\Users\\Tom Rishel\\OneDrive - The University of Southern Mississippi\\Documents\\Research\\TKX\\twitter_screen_names_by_country", 
    #                           "C:\\Users\\Tom Rishel\\OneDrive - The University of Southern Mississippi\\Documents\\Research\\TKX\\arabic_stopword_list.txt")
    #ArabTweetTransformCorpus()
    #ArabTweetSimilarities()
    #files2Docs(sys.argv[1])

