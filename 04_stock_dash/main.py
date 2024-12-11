from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from io import BytesIO
import json

# OS libs
import os
from dotenv import load_dotenv


# FastAPI application setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



alpha_vantage_api_key = os.getenv("alpha_vantage_api_key")



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the homepage with a form to input the stock symbol."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/stock", response_class=HTMLResponse)
async def get_stock_graph(request: Request, symbol: str = Form(...)):
    """
    Fetch stock data and generate a Plotly graph for the given symbol.
    """
    print(f"Received ticker: {symbol}")  # Debugging line

    try:
        # Fetch stock data
        stock_consolidate_df = fetch_stock_data(symbol)

        # Create Plotly graph
        plotly_price_EPS_html = plotly_price_EPS_graph(stock_consolidate_df)
        plotly_pe_ttm_avg_html = plotly_pe_ttm_avg_graph(stock_consolidate_df)
        # Return graph to user
        return templates.TemplateResponse(
            "graph.html",
            {
                "request": request
                ,"plotly_price_EPS_html": plotly_price_EPS_html
                ,"plotly_pe_ttm_avg_html": plotly_pe_ttm_avg_html
                ,"symbol": symbol

                },
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})



def fetch_stock_data(symbol):
    """
    Fetch and process stock data for a given symbol.
    """

    # STOCK SPLIT FACTOR section
    url = f'https://www.alphavantage.co/query?function=SPLITS&symbol={symbol}&apikey={alpha_vantage_api_key}'
    r = requests.get(url)
    data = r.json()

    for key, value in data.items():
        if key == 'data':
            if len(value) > 0:
                stock_split_record_df = pd.DataFrame(value)
                stock_split_record_df['split_factor'] = pd.to_numeric(stock_split_record_df['split_factor'], errors='coerce') # change split_factor series to numeric data
                stock_split_record_df['effective_date'] = pd.to_datetime(stock_split_record_df['effective_date'])
            else:
                stock_split_record_df = pd.DataFrame()
                stock_split_record_df['split_factor'] = 1
                stock_split_record_df['effective_date'] = datetime.today()


    # Daily quote section
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={alpha_vantage_api_key}&outputsize=full'
    r = requests.get(url)
    data = r.json()

    for key, value in data.items():
        if key == 'Time Series (Daily)':

            selected_cols = [
                '4. close'
            ]

            Daily_stock_df = pd.DataFrame(value).transpose()[selected_cols] # tranpose the dataframe and sub select selected cols

            # Rename columns
            Daily_stock_df.rename(
                columns={
                    '4. close': 'stock_price'
                    }
                ,inplace=True
                )
            
            Daily_stock_df["stock_price"] = Daily_stock_df["stock_price"].astype(str).apply(lambda x: float(x))
            Daily_stock_df["stock_price"] = Daily_stock_df["stock_price"].round(2)
            Daily_stock_df.index = pd.to_datetime(Daily_stock_df.index)


    # split the stock price if applied
    for date_i in Daily_stock_df.index.date:
        for date_j in stock_split_record_df['effective_date'].dt.date:
            if date_i == date_j:

                # stock price to divided the split factor
                Daily_stock_df.loc[Daily_stock_df.index.date < date_j, 'stock_price'] /= (stock_split_record_df['split_factor'][stock_split_record_df['effective_date'].dt.date == date_j].values[0])




    # Earning section
    # past earnings from alpha vintage API
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={alpha_vantage_api_key}'
    r = requests.get(url)
    data = r.json()

    for key, value in data.items():
        if key == 'annualEarnings':

            selected_cols = [
                'fiscalDateEnding'
                ,'reportedEPS'
            ]

            annualEPS_df = pd.DataFrame(value) # tranpose the dataframe and sub select selected cols


            annualEPS_df['fiscalDateEnding'] = pd.to_datetime(annualEPS_df['fiscalDateEnding']).dt.year


            # Convert the column to decimal type
            for col in selected_cols:
                if col in ['reportedEPS']:
                    annualEPS_df[f'{col}'] = annualEPS_df[f'{col}'].astype(str).apply(lambda x: float(x))

                else:
                    continue
            
            # clean annualEPS_df
            annualEPS_df = annualEPS_df.sort_values('reportedEPS', ascending=False).drop_duplicates('fiscalDateEnding')
            annualEPS_df = annualEPS_df.sort_values('fiscalDateEnding', ascending=False).reset_index(drop=True)



        if key == 'quarterlyEarnings':

            selected_cols = [
                'reportedDate'
                ,'reportedEPS'
            ]

            qtrEPS_df = pd.DataFrame(value)[selected_cols] # tranpose the dataframe and sub select selected cols
            qtrEPS_df['reportedDate'] = pd.to_datetime(qtrEPS_df['reportedDate'])

            # Convert the column to decimal type
            for col in selected_cols:
                if col in ['reportedEPS']:
                    qtrEPS_df[col] = qtrEPS_df[col].astype(str).apply(lambda x: float(x) if x not in [None, 'None', 'nan', 'NaN'] else float(0))
                else:
                    continue



    # Consolidated section
    stock_consolidate_df = Daily_stock_df
    stock_consolidate_df_date = stock_consolidate_df.index
    for i in stock_consolidate_df_date:
                
        # Filter the DataFrame to include only dates(index) less than or equal to the target date
        filtered_qtrEPS_df = qtrEPS_df[qtrEPS_df['reportedDate'] < i]

        # Select the first four rows from the past_qtrs_EPS
        past_4_qtrs_EPS = filtered_qtrEPS_df.head(4) 
        past_3_qtrs_EPS = filtered_qtrEPS_df.head(3)
        past_1_qtr_EPS = filtered_qtrEPS_df.head(1)

        # Calculate the sum of the numeric values in the selected rows
        EPS_TTM = past_4_qtrs_EPS['reportedEPS'].values.sum()
        EPS_curr_qtr = past_1_qtr_EPS['reportedEPS'].values.sum()

        # assign each index row with the EPS_TTM
        stock_consolidate_df.loc[i, "EPS_TTM"] = EPS_TTM
        stock_consolidate_df.loc[i, "EPS_currentQtr"] = EPS_curr_qtr


    # stock's stats
    stock_consolidate_df["Ticker"] = symbol
    stock_consolidate_df["PE_TTM"] = (stock_consolidate_df["stock_price"] / stock_consolidate_df["EPS_TTM"]).round(2)
    stock_consolidate_df["PE_TTM_avg"] = stock_consolidate_df["PE_TTM"].mean().round(2)
    stock_consolidate_df["PE_TTM_std"] = np.std(stock_consolidate_df["PE_TTM"]).round(2)
    stock_consolidate_df["PE_TTM_volatility_+"] = (stock_consolidate_df["PE_TTM_avg"] + stock_consolidate_df["PE_TTM_std"]).round(2) # 这个是PE的波动范围上限
    stock_consolidate_df["PE_TTM_volatility_-"] = (stock_consolidate_df["PE_TTM_avg"] - stock_consolidate_df["PE_TTM_std"]).round(2) # 这个是PE的波动范围下限

    stock_consolidate_df["relative_valuation_TTM_+"] = (stock_consolidate_df["PE_TTM_volatility_+"] * stock_consolidate_df["EPS_TTM"]).round(2) # 这个是relative valuation的价格上限
    stock_consolidate_df["relative_valuation_TTM_-"] = (stock_consolidate_df["PE_TTM_volatility_-"] * stock_consolidate_df["EPS_TTM"]).round(2) # 这个是relative valuation的价格下限
    stock_consolidate_df["relative_valuation_TTM_median"] = (np.median([stock_consolidate_df["relative_valuation_TTM_+"][0], stock_consolidate_df["relative_valuation_TTM_-"][0]])).round(2) #这个是根据最新TTM PE估值的价格中位数

    # if stock_consolidate_df.empty:
    #     raise ValueError(f"No stock data available for symbol {symbol}.")
    
    return stock_consolidate_df



def plotly_price_EPS_graph(stock_consolidate_df):
    """
    Generate a Plotly price EPS TTM graph from the stock DataFrame.
    """

    if stock_consolidate_df.empty:
        raise ValueError("DataFrame is empty, cannot generate graph.")

    # Create the figure
    fig = go.Figure()

    # Add stock_price on primary y-axis (left)
    fig.add_trace(go.Scatter(
        x=stock_consolidate_df.index,
        y=stock_consolidate_df["stock_price"],
        mode='lines',
        line=dict(color='black'),
        name='Stock Price',
        yaxis="y1"
    ))

    # Add EPS_TTM on secondary y-axis (right)
    fig.add_trace(go.Scatter(
        x=stock_consolidate_df.index,
        y=stock_consolidate_df["EPS_TTM"],
        mode='lines',
        fill='tonexty',  # Shadow effect
        line=dict(color='green'),
        name='EPS TTM',
        yaxis="y2"
    ))


    # Update layout to remove grid and configure dual y-axes
    fig.update_layout(
        title="Stock Metrics Over Time",
        xaxis=dict(
            title=None
            ,showgrid=False
            ),
        yaxis=dict(
            title=None,
            showgrid=False,
            titlefont=dict(color="black"),
            tickfont=dict(color="black")
        ),
        yaxis2=dict(
            title=None,
            overlaying="y",  # Overlay with y1
            side="right",
            showgrid=False,
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue")
        ),
        legend_title="Metrics",
        template="plotly_white",
        plot_bgcolor='white'  # Background color
    )
    # Convert Plotly graph to HTML
    graph_html = fig.to_html(full_html=False)

    return graph_html



def plotly_pe_ttm_avg_graph(stock_consolidate_df):
    """
    Generate a PE TTM statics graph from the stock DataFrame.
    """

    if stock_consolidate_df.empty:
        raise ValueError("DataFrame is empty, cannot generate graph.")
    
    # Create the figure
    fig = go.Figure()

    # Add PE_TTM on primary y-axis
    fig.add_trace(go.Scatter(
        x=stock_consolidate_df.index,
        y=stock_consolidate_df["PE_TTM"],
        mode='lines',
        line=dict(color='black'),
        name='PE TTM',
    ))

    # Add PE_TTM_avg as a horizontal line
    fig.add_trace(go.Scatter(
        x=stock_consolidate_df.index,
        y=[stock_consolidate_df["PE_TTM_AVG"][0]] * len(stock_consolidate_df.index),
        mode='lines',
        line=dict(color='green', dash='dash'),
        name='Avg.PE TTM',
    ))

    # Update layout
    fig.update_layout(
        title="PE TTM Over Time",
        xaxis=dict(
            title=None,
            showgrid=False,
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
        ),
        legend_title="Metrics",
        template="plotly_white",
        plot_bgcolor='white',  # Background color
    )

    # Convert Plotly graph to HTML
    graph_html = fig.to_html(full_html=False)

    return graph_html