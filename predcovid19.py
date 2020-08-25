# -*- coding: utf-8 -*-
"""
Covid Predictions App 

@author: mariano 
"""

#-------------------------------
# Import libraries
#-------------------------------

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta



#######################################
# Define global variables
#######################################

# Define number of days considered for cumulative number
Cum_Sum_Days = 7

# Define Cum_Cases as sort name for Cum Sum
Cum_Cases = Cum_Sum_Days
    

#######################################
# Define functions
#######################################

#------------------------------
# function to load data 
#------------------------------


def load(Country):
    
    url = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx"

    df = pd.read_excel(url,sheet_name='COVID-19-geographic-disbtributi')

    #------------------------------
    # Prepare data 
    #-------------------------------

    # Change names of some large colum names
    df = df.rename(columns={"countriesAndTerritories": "countries", 
                            "countryterritoryCode":"Country_Code",
                            "continentExp":"continent",
                            "Cumulative_number_for_14_days_of_COVID-19_cases_per_100000":"Cum_Cases_14d",
                           })
    
    df.fillna(0,inplace=True)

    # Transform date to the right format
    df['date'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')

    # Dropput the unneded columns
    df.drop('dateRep', axis=1, inplace=True)
    
    
    #--------------------------------------
    # Add rolling to all the countries
    #--------------------------------------
 
    # Sort by date all dataset to make the cumsum correctly
    df.sort_values(by = 'date', inplace=True)
    
    # Select country
    df_covid = df[df['countries'] == Country]

    # get the rolling sum of cases for last 14 days
    df_covid['rolling']=df_covid.cases.rolling(Cum_Sum_Days).sum()
    
    
    # Drop na
    df_covid = df_covid.dropna()
    
    return df_covid

    
#-------------------------------
# Function to train model
#-------------------------------   

def model_covid(df_covid):
    
    # Fix random seed for reproducibility
    np.random.seed(7)
    
    
    # Use the dataframe from here to build machine learning model
    #data = df_covid.copy()
    
    # Use the rolling column as timeseries
    dataset = df_covid[['rolling']]
    dataset = dataset.values

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Split into train and test sets
    train_size = int(len(dataset) * 0.90)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    # Convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        
       dataX, dataY = [], []
       for i in range(len(dataset)-look_back):
	        	a = dataset[i:(i+look_back), 0]
	        	dataX.append(a)
	        	dataY.append(dataset[i + look_back, 0])
	       
       return np.array(dataX), np.array(dataY)
    
    # Reshape into X=t and Y=t+1
    look_back = 7
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


    #---------------------------
    # Train model 
    #---------------------------

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(3, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=5, verbose=0)
    
    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)


    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    
    #--------------------------------------
    # Extract last value of the testing 
    #--------------------------------------

    # Show the last element predicted  
    testPredict_length = len(testPredict)
    Pred_last_element = testPredict[testPredict_length - 1]

    # Show the last real element  testY
    Flat_TestY = testY.flatten()
    testY_length = len(Flat_TestY)
    Real_last_element = Flat_TestY[testY_length-1]

    #-------------------------------------------------
    # Pred new data sample
    #-------------------------------------------------

    # Use original data 
    #data_pred = df_covid.copy()

    # Get the last elements to be used to make new prediction
    data_pred_os = df_covid[-look_back:]

    # Use the rolling column as timeseries
    dataset_os = data_pred_os[['rolling']]
    dataset_os = dataset_os.values

    # Normalize the dataset
    dataset_os = scaler.transform(dataset_os)

    # Reshape input to be [samples, time steps, features]
    dataset_os = np.reshape(dataset_os, (dataset_os.shape[1], dataset_os.shape[0], 1))

    # Make predictions
    dt_pred = model.predict(dataset_os)

    # Invert predictions
    dt_pred = scaler.inverse_transform(dt_pred)
    
    
    #----------------------------------------------------------
    # Save last prediction into a file for further anlysis
    #----------------------------------------------------------

    # Get date of last current value
    Current_Date = df_covid.iloc[len(df_covid)-1,df_covid.columns.get_loc("date")]

    # Tomorrow date 
    Tomorrow_Date = Current_Date + timedelta(days=1)

    # Pass data to dataframe 
    data = {'Real_Point'      : [int(Real_last_element)],
            'Predicted_Point' : [int(Pred_last_element[0])],
            'Pred_Tomorrow'   : [int(dt_pred[0,0])],
            'Date'            : Current_Date,
            'Tomorrow_Date'   :Tomorrow_Date}

    df_pred = pd.DataFrame(data, columns = ['Date','Real_Point', 
                                            'Predicted_Point', 
                                            'Tomorrow_Date', 
                                            'Pred_Tomorrow'])
    
    return df_pred


#######################################
# Create a dash website application 
#######################################

#-----------------------------
# Call the functions 
#-----------------------------

def run_covid(Country):
    
    df_covid = load(Country)
    
    df = model_covid(df_covid)
    
    return df

#-----------------------------
# Load Data
#-----------------------------

url = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx"

data = pd.read_excel(url,sheet_name='COVID-19-geographic-disbtributi')

countries = data.countriesAndTerritories.unique()


#-----------------------------
# Define app dashboard
#-----------------------------

children_text_H1 = "Covid Prediction in Country"
subtitle = "Covid 19 prediction using LSTM networks. Prediction over cumulative cases."

children_text_H2 = "Select the country to make predictions"

children_text_H3 = "Click the buttom bellow to make predictions over the country selected"

title1 = "Covid 19 prediction for last updated day:"
title2 = "Covid 19 prediction for day"


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    
    html.H1(id='H1',children=children_text_H1),

    html.Div(children= 
        subtitle
    ),
    
     html.H3(children= 
        children_text_H2
    ),
    
    html.Hr(),
    
    dcc.Dropdown(
        id = "input",
        options=[{'label': i, 'value': i} for i in countries],
        value=''
    ),
      
    html.Div(id='dd-output-container'),
    
    html.Hr(),
    
         html.H3(children= 
        children_text_H3
    ),
    
    html.Button('Click to predict', id='button'),
    html.H3(id='button-clicks'),

    html.Hr(),
   
    dcc.Graph(id='graph1'),
    
    
    dcc.Graph(id='graph2'),
   


])



@app.callback([Output('graph1', 'figure'),
              Output('graph2', 'figure'),
              Output('H1', 'children')],
              [Input('button', 'n_clicks'),
               Input('input', 'value')])

def update_figure(n_clicks,value):
    
    if n_clicks > 0:
       df = run_covid(value)

     
     
    return  {
                    'data': [
                       {'x': [1], 'y': [df.Predicted_Point[0]], 'type': 'bar', 'name': 
                        'Prediction'},
                       {'x': [2], 'y': [df.Real_Point[0]], 'type': 'bar', 'name': 
                        u'Real'},
                      ],
                    'layout': {
                      'title': "Covid 19 prediction for {} on: {}".format(value,df.Date[0]) 
                    }
            } ,  {
                    'data': [
                        {'x': [1], 'y': [df.Pred_Tomorrow[0]], 'type': 'bar', 'name': 
                         'Prediction_Tomorrow'}
                      ],
                    'layout': {
                      'title': "Covid 19 prediction for {} on day: {}".format(value,df.Tomorrow_Date[0])
                    }
            } , "Covid Prediction in Country: {}".format(value)  
                



if __name__ == '__main__':
    app.run_server(debug=False,use_reloader=False, threaded=False)
    

