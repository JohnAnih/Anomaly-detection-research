import pandas as pd
import numpy as np
import urllib.request
from influxdb import InfluxDBClient
import time


def get_db_ready(db_name='water_data'):
    
    # Make a connection
    client = InfluxDBClient(host='localhost', 
                            port=8086, 
                            username='root', 
                            password='root', 
                            database=db_name)
    
    # drop the database to avoid duplicates if it already exist
    client.drop_database(db_name)
    
    # Create a database called "WaterTrainingdata"
    client.create_database(db_name)
    
    # Confirm that the database data "WaterTrainingdata" exist
    print(client.get_list_database())
    
    # Make use of the database
    client.switch_database(db_name)
    
    return client



def write_points_to_influxdb(X_test, y_test, verbose=True, db_name='water_data'):
    
    client = get_db_ready(db_name=db_name)
    
    X_test['EVENT'] = y_test
    
    df = X_test
    
    for index, cols in df.iterrows():
        json_body = [
            {
                "measurement" : "water_data",
                "time":index,
                "fields": {
                    "Tp": cols[0],
                    "Cl": cols[1],
                    "pH": cols[2],
                    "Redox": cols[3],
                    "Leit": cols[4],
                    "Trueb": cols[5],
                    "Cl_2": cols[6],
                    "Fm": cols[7],
                    "Fm_2": cols[8],
                    "EVENT": cols[9]
                    }
                }
            ]
        
        if verbose:
            print(json_body)
            
            
        client.write_points(json_body)
    
    url = 'http://localhost:9092/kapacitor/v1/tasks/{}/httpout'.format(db_name)
      
    return url

    print('Data points successfully written')
    


def transform_to_dataframe(json):
    
    json_data = pd.read_json(json)['series'][0]['values']

    df = pd.DataFrame(json_data,
                      
                      columns = ['time' , 
                                 'Cl', 
                                 'Cl_2',
                                 'EVENT',
                                 'Fm',
                                 'Fm_2',
                                 'Leit',
                                 'Redox',
                                 'Tp',
                                 'Trueb',
                                 'pH'])

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    return df