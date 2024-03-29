#  Modules imports
import streamlit as st
import pandas as pd
import numpy as np
import math
import datetime

from bs4 import BeautifulSoup
import re
import os
import glob
import requests
from urllib.request import urlopen
import holidays

from prophet import Prophet
from prophet.serialize import model_from_json

import ephem
import matplotlib.pyplot as plt

from helper_functions import data_loader, signal, day_night, data_formater, future_to_model
st.set_option('deprecation.showPyplotGlobalUse', False)

def main()
    st.write("""
        # 
        """)
    st.markdown("<h1 style='text-align: center;'> Electricity Price Forcaster  </h1>", unsafe_allow_html=True)

    st.write('---')





if __name__ == "__main__":
    main()
