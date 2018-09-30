from urllib3.exceptions import ProtocolError, ReadTimeoutError
import tweepy
import dataset
import json
from tweepy import StreamListener
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from models import *
import pandas as pd
import numpy as np
from config import *

engine = create_engine('postgresql://paulinazheng:@localhost5432/flu')
Session = sessionmaker(bind=engine)
session = Session()

analyser = SentimentIntensityAnalyzer()

def sentiment_score(text):
    return analyser.polarity_scores(text)

api = tweepy.API(auth_r)

def calculate_centroid(box):
    avg_lat = (box[1][1] + box[0][1])/2
    avg_long = (box[2][0] + box[1][0])/2
    return avg_lat, avg_long

LOCATION = [-164.639405, 18.776344, -66.947028, 71.76871]

cities = [('New York', 40.7127837, -74.00594129999999),
 ('Los Angeles', 34.0522342, -118.24368490000002),
 ('Chicago', 41.8781136, -87.62979820000001),
 ('Houston', 29.7604267, -95.36980279999999),
 ('Philadelphia', 39.9525839, -75.1652215),
 ('Phoenix', 33.4483771, -112.07403729999999),
 ('San Antonio', 29.4241219, -98.4936282),
 ('San Diego', 32.715738, -117.1610838),
 ('Dallas', 32.7766642, -96.7969879),
 ('San Jose', 37.338208200000004, -121.88632859999998),
 ('Austin', 30.267153000000004, -97.7430608),
 ('Indianapolis', 39.768403, -86.158068),
 ('Jacksonville', 30.3321838, -81.65565099999998),
 ('San Francisco', 37.7749295, -122.4194155),
 ('Columbus', 39.9611755, -82.99879419999998),
 ('Charlotte', 35.2270869, -80.8431267),
 ('Fort Worth', 32.7554883, -97.3307658),
 ('Detroit', 42.331427000000005, -83.0457538),
 ('El Paso', 31.7775757, -106.44245590000001),
 ('Memphis', 35.1495343, -90.0489801),
 ('Seattle', 47.6062095, -122.33207079999998),
 ('Denver', 39.739235799999996, -104.990251),
 ('Washington', 38.9071923, -77.03687070000001),
 ('Boston', 42.360082500000004, -71.0588801),
 ('Nashville-Davidson', 36.1626638, -86.78160159999999),
 ('Baltimore', 39.2903848, -76.6121893),
 ('Oklahoma City', 35.4675602, -97.5164276),
 ('Louisville/Jefferson County', 38.252664700000004, -85.7584557),
 ('Portland', 45.523062200000005, -122.67648159999999),
 ('Las Vegas', 36.169941200000004, -115.13982959999998)]

def add_item(item):
    db.session.add(item)
    db.session.commit()

def find_closest_city(centroid_lat, centroid_long, cities=cities):
    smallest = 10000
    point = (centroid_lat, centroid_long)
    for city in cities:
        dist = np.sqrt((city[1]-point[0])**2 + (city[2]-point[1])**2)
        if dist < smallest:
            smallest = dist
            closest = city
    return closest

def get_city_id(lat, long):
    closest = find_closest_city(lat, long, cities=cities)
    if closest[0] not in [city.name for city in City.query.all()]:
        city = City(name=closest[0], lat=closest[1], long=closest[2])
        add_item(city)
        city_id = city.id
    else:
        city = City.query.filter_by(name = closest[0]).all()
        city_id=city[0].id
    return city_id

def get_or_create_user(user_id, location):
    user = User.query.filter_by(user_id=user_id).first()
    if user:
        return user
    else:
        user = User(user_id=user_id, location=location)
        add_item(user)
        return user

def get_or_create_tweet(user_id, location, twitter_id, created, centroid_lat, centroid_long, text, city_id):
    tweet = Tweet.query.filter_by(twitter_id=twitter_id).first()
    if tweet:
        return tweet
    else:
        user = get_or_create_user(user_id, location)
        sentiment = sentiment_score(text)
        positivity = round(sentiment['pos'], 4)
        negativity = round(sentiment['neg'], 4)
        compound = round(sentiment['compound'], 4)
        polarity = round((TextBlob(text)).sentiment.polarity, 4)
        tweet = Tweet(twitter_id=twitter_id, text=text, created=created, centroid_lat=centroid_lat,
        centroid_long=centroid_long, positivity=positivity, negativity=negativity, compound=compound,
        polarity=polarity, user_id=user.id, city_id=city_id)
        add_item(tweet)
        return tweet

class StreamListener(tweepy.StreamListener):
    def on_connect(self):
        print("Now we're saving from Twitter!")

    def on_status(self, status):
    #avoids retweets, non-geolocated
        if status.retweeted:
            return
        if not status.place:
            return
        if status.lang == 'en':
            if status.truncated == True:
                text = status.extended_tweet['full_text']
                if len(text) > 320:
                    text = text[:320]
            else:
                text=status.text
            id_str = status.id_str
            created = status.created_at
            box = status.place.bounding_box.coordinates[0]
            centroid_lat, centroid_long = calculate_centroid(box)
            coords = status.coordinates
            if coords is not None:
                coords = json.dumps(coords)
            loc = status.user.location
            user_id = str(status.user.id)
            sentiment = sentiment_score(text)
            positivity = round(sentiment['pos'], 4)
            negativity = round(sentiment['neg'], 4)
            compound = round(sentiment['compound'], 4)
            polarity = round((TextBlob(text)).sentiment.polarity, 4)
            city_id = get_city_id(centroid_lat, centroid_long)
            get_or_create_tweet(user_id, loc, id_str, created, centroid_lat, centroid_long, text, city_id)

    def on_exception(self, exception):
           print(exception)
           return

    def on_error(self, status_code):
        if status_code == 420:
            return False

flu = ['flu', 'influenza', 'cough', 'fever', 'sore throat', 'headache',
        'phlegm', 'runny nose', 'stuffy nose', 'Robitussin',
        'dayquil', 'nyquil', 'tamiflu', 'vomit', 'body ache', 'mucinex',
        'pneumonia', 'vomit', 'bodyache', 'medicine']

stream_listener = StreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
while True:
    try:
        stream.filter(track=flu)
    except (ProtocolError, AttributeError, ReadTimeoutError):
        continue
test = stream.filter(track=flu)
