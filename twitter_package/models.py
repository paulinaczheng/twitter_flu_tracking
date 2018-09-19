from  __init__ import *

class Tweet(db.Model):
    __tablename__ = 'tweets'
    id = db.Column(db.Integer, primary_key=True)
    twitter_id = db.Column(db.String(30))
    text = db.Column(db.String(320))
    created = db.Column(db.DateTime)
    retweets = db.Column(db.Integer)
    centroid_lat = db.Column(db.Float)
    centroid_long = db.Column(db.Float)
    positivity = db.Column(db.Float)
    negativity = db.Column(db.Float)
    compound = db.Column(db.Float)
    polarity = db.Column(db.Float)
    status = db.Column(db.Integer) #0 for not relevant, 1 for relevant
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    user = db.relationship('User', back_populates='tweets')
    city_id = db.Column(db.Integer, db.ForeignKey('cities.id'))
    city = db.relationship('City', back_populates='tweets')

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(200))
    created = db.Column(db.DateTime)
    description = db.Column(db.String(200))
    location = db.Column(db.String(200))
    followers = db.Column(db.Integer)
    friends = db.Column(db.Integer)
    statuses = db.Column(db.Integer)
    tweets = db.relationship('Tweet', back_populates='user')
    city_id = db.Column(db.Integer, db.ForeignKey('cities.id'))
    city = db.relationship('City', back_populates='users')

class City(db.Model):
    __tablename__ = 'cities'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(350))
    lat = db.Column(db.Float)
    long = db.Column(db.Float)
    users = db.relationship('User', back_populates='city')
    tweets = db.relationship('Tweet', back_populates='city')

db.create_all()
