import pandas as pd
import numpy as np
import urllib.request
from sklearn.metrics import classification_report
from influxdb import InfluxDBClient
import time

from general_funcs import (interpolate_df, 
                           get_stationary_df)


def get_db_ready(db_name='water_data'):
    """
    get_db_ready makes a connection with influxDB

    This function would work when your influxDB is open and active 
    Otherwise the connection would be refused.
    
    Learn more here: https://docs.influxdata.com/influxdb/v1.8/query_language/explore-schema/

    Parameters
    ----------
    db_name : str, optional
        [description], by default 'water_data'

    Returns
    -------
    database
        created database
    """    ""
    
    client = InfluxDBClient(host='localhost', 
                            port=8086, 
                            username='root', 
                            password='root', 
                            database=db_name)
    
    # drop the database to avoid duplicates if it already exist
    client.drop_database(db_name)
    
    # Create a database called "WaterTrainingdata"
    client.create_database(db_name)
    
    # Make use of the database
    client.switch_database(db_name)
    
    return client



def write_points_to_influxdb(X_test, y_test, verbose=True, db_name='water_data'):
    """
    write_points_to_influxdb makes a connection and creates a database with the database name specified

    Parameters
    ----------
    X_test : DataFrame/ Numpy arrays
        inputs
    y_test : DataFrame/ Numpy arrays
        outputs
    verbose : bool, optional
        option to show print messages, by default True
    db_name : str, optional
        Name of database, by default 'water_data'

    Returns
    -------
    str
        url where the data is uploaded for further retrieval
    """    ""
    
    client = get_db_ready(db_name=db_name)
    
    X_test['EVENT'] = y_test
    
    df = X_test
    
    for index, cols in df.iterrows():
        json_body = [
            {
                "measurement" : db_name,
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
        
    print('Data points successfully written')
    
    url = 'http://localhost:9092/kapacitor/v1/tasks/{}/httpout'.format(db_name)
      
    return url

    
    


def transform_to_dataframe(url):
    """
    transform_to_dataframe transforms the data into a readable pandas DataFrame

    Parameters
    ----------
    url : str
        url where the json data is located

    Returns
    -------
    DataFrame
        Pandas DataFrame
    """    ""
    
    json_data = pd.read_json(url)['series'][0]['values']

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



def automate_data_process(model, inputs, output, target='EVENT'):
    """
    automate_data_process automates the data pipeline process by:
    >> Checking for new data points every minute
    >> Transforming the data into a dataframe
    >> Performing the data preparation steps and making predictions
    >> Finally evaluates the models

    Parameters
    ----------
    model : Model
        Finalised model
    inputs : DataFrame/Numpy arrays
        Inputs
    output : DataFrame/Numpy arrays
        Outputs
    target : str, optional
        Name of the output variable, by default 'EVENT'
    """    ""
    
    while True:
        url = write_points_to_influxdb(X_test=inputs, y_test=output)
        recent_1 = urllib.request.urlopen(url).read()
        recent_2 = urllib.request.urlopen(url).read()
        
        while recent_1 == recent_2:
            print("No New Data")
            
            time.sleep(60)
            
            recent_2 = urllib.request.urlopen(url).read()

            data = transform_to_dataframe(recent_2)

            x= data.drop(target, axis=1)

            y_true= data[target]

            # fill missing values by linear interpolation
            x = interpolate_df(df=x)

            # make the time series stationary
            x = get_stationary_df(df= x)

            y_pred = model.predict(x)

            print(classification_report(y_true, y_pred))