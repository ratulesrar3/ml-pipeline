import psycopg2
import numpy as np
import pandas as pd
import json
import logging


class Connecc():
    '''
    Establish connection to Database
    Parameters: filepath to json file containing credentials for db
    '''
    def __init__(self, filepath='config.json'):
        self.conn = None
        self.cur = None
        self.config = filepath

    def load_config(self):
        '''
        Loads db info from config file
        '''
        with open(self.config, 'r') as f:
            config = json.load(f)
        return config

    def set_connection(self):
        '''
        Takes a db config file and returns a connection object
        '''
        config = self.load_config()

        try:
            self.conn = psycopg2.connect(**config)
            print('Connecc')
        except:
            print('Error connecting to database')
        finally:
            self.conn.set_session(autocommit=True)
            self.cur = self.conn.cursor()

    def close_connection(self):
        '''
        Terminates the connection
        '''
        try:
            self.cur.close()
            self.conn.close()
            print('Closed connection')
        except:
            print('Error closing the connection')

    def execute_query(self, query):
        '''
        tries executing query, rolls back bad query
        '''
        try:
            self.cur.execute(query)
        except:
            print("Roll Bacc")
            logging.exception("Bad Query")
            self.conn.rollback()

    def get_df(self, query):
        '''
        stores query output in pandas dataframe
        '''
        try:
            return pd.read_sql(query, self.conn)
        except:
            print("Roll Bacc")
            logging.exception("Bad Query")
            self.conn.rollback()