# Python script to download Arabic language tweets from identified Arabic twitter accounts
# USM TKX project, Summer, 2017
# Tom Rishel
#
# In order to use this script you must have:
# * Twitter API credentials - my API credentials are hard-coded below
# * A plain text file containing the screen names of the twitter accounts 
#   from which you wish to download tweets. Screen names should be one per 
#   line with no other text in the file.
# * A properly configured Python 3.6 or higher environment with the necessary libraries 
#   installed
# * The text file of screen names should be placed into the same directory as 
#   this script
# * To run the script from the command line "python tweetsToFilesByNameByMonth.py <filename>"
#   where <filename> is replaced by the name of the text file containing the
#   screen names
# * If you want to store tweets into a database, a properly configured database
# 
# The output of this script is either: 
# 1. A set of files containing tweets by each screen name
# 	separated by the month in which the tweet was created. A separate file will be
# 	created for each month in which tweets were created by each screen name.
# AND
# 2. A pre-defined database populated with the same tweets. 

#!/usr/bin/env python
# encoding: utf-8

import tweepy #https://github.com/tweepy/tweepy
import sys #used to get command line arguments
import pymysql #used to manage connection to mysql database
import pymysql.cursors #used to manage cursors for database
import time #used to handle time objects
from datetime import date #used to handle date objects
import os

#Twitter API credentials
consumer_key = "your twitter api consumer key"
consumer_secret = "your twitter api consumer secret"
access_key = "your twitter api access key"
access_secret = "your twitter api access secret"

#Create connection to mysql database
#cnx = pymysql.connect(user='your mysql user name', password='your mysql password',
#                        host='ip address of your database server',
#                        database='your database to store the tweets')

#function to store the most recent 3240 tweets for a given twitter screen name in files divided by month		
def get_all_tweets(screen_name, path, filename):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	
	#api = tweepy.API(auth)
	# change tweepy initialization to wait on rate limit
	api = tweepy.API(auth, wait_on_rate_limit=True)
	
	#create a cursor object to store data for the database
#	cursor = cnx.cursor()
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200, tweet_mode='extended')
	
	#save most recent tweets
	alltweets.extend(new_tweets)
	
	if len(alltweets) <= 0:
		return

   	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	
	#save the month and year of the most recent tweet
	new_tweet_month = padMonth(alltweets[0].created_at.month)
	new_tweet_year = alltweets[0].created_at.year
	tweet_month = new_tweet_month
	tweet_year = new_tweet_year

	#create and open a text file to store tweets for this user, month, and year
	f = open('%s_%s_%s_%s.txt' % (path + "tweets\\" + filename[0:-4], screen_name, tweet_year, tweet_month), 'wb')


#################################################################################################################################################################################################################################
# Use the code below to write tweets to the connected database and to the files simultaneously
#################################################################################################################################################################################################################################
		
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print("getting tweets before %s" % (oldest))
		
		#loop through tweets inserting them into the file
		for oneTweet in new_tweets:

			#create a new file for each new month
			new_tweet_month = padMonth(oneTweet.created_at.month)
			new_tweet_year = oneTweet.created_at.year
			if new_tweet_month != tweet_month or new_tweet_year != tweet_year:
				f.close()
				tweet_month = new_tweet_month
				tweet_year = new_tweet_year
				f = open('%s_%s_%s_%s.txt' % (path + "tweets\\" + filename[0:-4], screen_name, tweet_year, tweet_month), 'wb')
						
			#create and open a text file to store tweets for this user, month, and year
			#f = open('%s_%s_%s_%s.txt' % (filename[0:-4], screen_name, tweet_year, tweet_month), 'wb')
	
			#SQL to add tweet to table
#			add_tweet = ("INSERT INTO arabic_tweets (tweet_id, screen_name, country, created_on, text) VALUES (%s, %s, %s, %s, %s)") 

			#only add tweets that have text
			try:
				#try to store the text field from the tweet
				if len(oneTweet.text) > 0:
#					try:
#						tweet_data = (oneTweet.id_str.encode("utf-8"), str(screen_name).encode("utf-8"), str(filename[0:-4]).encode("utf-8"), str(oneTweet.created_at).encode("utf-8"), str(oneTweet.text.encode("utf-8")))
#						cursor.execute(add_tweet, tweet_data)
#						cnx.commit()
					
#					except(cnx.Error, cnx.Warning) as e:
#						print(e)
						#return None
					
					try:
						f.write(oneTweet.text.encode("utf-8"))
					except:
						pass

			except AttributeError:
				#if the tweet does not have a text field try to store the full_text field
				if len(oneTweet.full_text) > 0:
#					try:
#						tweet_data = (oneTweet.id_str.encode("utf-8"), str(screen_name).encode("utf-8"), str(filename[0:-4]).encode("utf-8"), str(oneTweet.created_at).encode("utf-8"), str(oneTweet.full_text.encode("utf-8")))
#						cursor.execute(add_tweet, tweet_data)
#						cnx.commit()

#					except(cnx.Error, cnx.Warning) as e:
#						print(e)
						#return None
					try:
						f.write(oneTweet.full_text.encode("utf-8"))
					except:
						pass
			# this code writes the ASCII encoded line ending character
			f.write(chr(10).encode("utf-8"))
			f.write(chr(13).encode("utf-8"))
			f.write(str("\n").encode("utf-8"))
												
		#all subsequent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		
		print("...%s tweets downloaded so far" % (len(alltweets)))
			
	pass
	#close the final file	
	f.close()			
#################################################################################################################################################################################################################################
						
#################################################################################################################################################################################################################################
# Use the code below to write tweets to files
#################################################################################################################################################################################################################################
#	#keep grabbing tweets until there are no tweets left to grab
#	while len(new_tweets) > 0:
#		print("getting tweets before %s" % (oldest))
#		
#		#loop through tweets inserting them into the database
#		for oneTweet in new_tweets:
#
#			#create a new file for each new month
#			new_tweet_month = padMonth(oneTweet.created_at.month)
#			new_tweet_year = oneTweet.created_at.year
#			if new_tweet_month != tweet_month or new_tweet_year != tweet_year:
#				tweet_month = new_tweet_month
#				tweet_year = new_tweet_year
#	
#
#			#only add tweets that have text
#			try:
#				#try to store the text field from the tweet
#				if len(oneTweet.text) > 0:
#					f.write(oneTweet.text.encode("utf-8"))
#	
#			except AttributeError:
#				#if the tweet does not have a text field try to store the full_text field
#				if len(oneTweet.full_text) > 0:
#					f.write(oneTweet.full_text.encode("utf-8"))
#			f.write(str("\n").encode("utf-8"))
#		
#		#all subsequent requests use the max_id param to prevent duplicates
#		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
#		
#		#save most recent tweets
#		alltweets.extend(new_tweets)
#		
#		#update the id of the oldest tweet less one
#		oldest = alltweets[-1].id - 1
#		
#		print("...%s tweets downloaded so far" % (len(alltweets)))
#			
#	pass
#	
#	#close the final file	
#	#f.close()			
#################################################################################################################################################################################################################################



#this function take a month and pads a zero to the front if needed
def padMonth(month):
	if month > 0 and month < 10:
		month = "0" + str(month)
	return month
		
#this function takes a filename containing twitter screen names, one per line, and runs the function above for each screen name
def file2tweets(path, filename):
	#open the file
	i = open(filename, 'r')
	#get a line
	for line in i:
		#get rid of any white space
		screen_name = line.strip()
		#check to make sure it is not a blank line
		if len(screen_name) > 0:
			print ("\ngetting tweets for %s\n" % (screen_name))
			
			#clean the filename if necessary
			#find the index of the last backslash
			index = filename.rfind("\\")
			#add one to get rid of the backslash itself
			filename = filename[index + 1:]
						
			#get the tweets and store the text in the file
			get_all_tweets(screen_name, path, filename)
	#close the files
	i.close()
	
	pass
		
def fileRunner(): #(directory)
    path = 'path to the folder that has subfolders with lists of accounts you want to harvest'
    countryFiles = os.listdir(path)
    for country in countryFiles:
        print(str(country[-4:]))
        if str(country)[-4:] == ".txt":
            fullPath = path + country
			# pass the path separately because we are going to use it later
            file2tweets(path, fullPath)


if __name__ == '__main__':
	fileRunner()
