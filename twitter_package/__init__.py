from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import dash

server = Flask(__name__)

server.config['DEBUG'] = True
server.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://paulinazheng@localhost:5432/flu"
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# server.config['SQLALCHEMY_ECHO'] = True

db = SQLAlchemy(server)

app = dash.Dash(__name__, server=server, url_base_pathname = '/')

from twitter_package.dashboard import *
# from twitter_package.config import *
