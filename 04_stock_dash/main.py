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



API_KEY = os.getenv("alpha_vantage_api_key")
# Parameters section
alpha_vantage_api_key = API_KEY


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the homepage with a form to input the stock ticker."""
    return templates.TemplateResponse("index.html", {"request": request})