from flask import Flask, json, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy
import requests

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

from datetime import datetime, timedelta
import yfinance as yf
from pandas_datareader import data as pdr

# This assign the app objective to the FLASK class
app = Flask(__name__)


#############################Section for Database development###################################
# This configure the backend database to sqlite, and create connection between sqlite to serverhost which is the app
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
# db = SQLAlchemy(app)


# # to create a stock class to store data
# # the stock class inheritage from the db.Model
# class Stock(db.Model):

#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(20), unique=True, nullable=False)
#     description = db.Column(db.String(50))

#     def __rep__(self):
#         return f'{self.name} - {self.description}'


# # create the database schema
# db.create_all()
#############################Section for Database development###################################

# Global parameters
default_start_date = datetime.now().date() - timedelta(days=7)
default_end_date = datetime.now().date()
requested_cols = [
    "Close"
    ,"Volume"
    ,"Volume % Change"
    ,"Close % Change"
    ]
metric_name_list = [
    'Volume'
    ,'Close'
]


def __get_stock_data():
    """
    This fucntion retreive the stock data using yfinance library and clean the dataframe with requested column
    """

    ticker_symbol = request.args.get(
        'ticker_symbol'
        ,default='MSFT'
        )
    
    search_start_date = request.args.get(
        'search_start_date'
        ,default=default_start_date
    )

    search_end_date = request.args.get(
        'search_end_date'
        ,default=default_end_date
        )


    # this will let pandas dataframe take over the reader format
    yf.pdr_override()

    # download dataframe
    df = pdr.get_data_yahoo(
        ticker_symbol.upper()
        ,start=search_start_date
        ,end=search_end_date
    )

    # create calculation
    df["Volume % Change"] = df["Volume"].pct_change() * 100
    df["Close % Change"] = df["Close"].pct_change() * 100

    # clean dataframe with requested column
    df = df[requested_cols]

    # fill the NA with 0
    df= df.fillna(0)

    # reset index columns, to let date col become a feature col
    df.reset_index(inplace=True)

    return df




# API route to the home page
@app.route("/")
def home():
	# we will use Flask's render_template method to render a website template.
    return render_template("index.html")


# API route for pulling the stock quote
@app.route('/quote')
def __get_stock_data_js():
    """
    This fucntion transfer the pandas dataframe into JSON format for web data ingestion
    """
    df = __get_stock_data()

    # transfer the pandas dataframe to a list of dictionaries for json serialize purpose
    df = df.to_dict(orient='records')
    
    return jsonify(df)


# API route for pulling the stock metrics
@app.route('/metrics')
def __period_metric_perc():
    """
    This function calculate period percentage change through all requested metrics
    """

    api1_response = requests.get('http://localhost:5000/quote')

    # Check if the request was successful (status code 200)
    if api1_response.status_code == 200:
        # Parse the JSON response from API 1
        df = api1_response.json()
        # You can now use api1_result in your second API logic

        # transfer the json to pandas dataframe
        df = pd.DataFrame(
            data=df
            ,columns=['Date'] + requested_cols
            )


        metric_value_list = []

        for metric in metric_name_list:
            period_metric_perc_delta = round(
                ((df[metric].iloc[-1] - df[metric].iloc[0]) / df[metric].iloc[0]) * 100, 4
            )
            metric_value_list.append(period_metric_perc_delta)

        metric_dict = dict(zip(metric_name_list, metric_value_list))

        return f'Summary of cumluateive changes from {default_start_date} to {default_end_date} is {metric_dict}'
    else:
        return jsonify({'error': 'Unable to get result from API 1'}), api1_response.status_code



@app.route('/line-chart')
def __create_line_chart():
    ticker_symbol = request.args.get('ticker_symbol', default='MSFT')
    search_start_date = request.args.get('search_start_date', default=default_start_date)
    search_start_date = request.args.get('search_end_date', default=default_end_date)

    stock_data = pd.DataFrame(__get_stock_data())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name=f'{ticker_symbol} Close Price'))

    # Update layout
    fig.update_layout(title=f'{ticker_symbol} Stock Close Price Over Time', xaxis_title='Date', yaxis_title='Close Price')

    return jsonify(fig.to_json())



# API route to the Metrics Summary page
@app.route('/metrics-summary')
def metrics_summary():
    return render_template("metrics_summary.html")


if __name__ == '__main__':
    app.run(debug=True)