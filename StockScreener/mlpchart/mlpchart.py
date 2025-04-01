import yfinance as yf

from mplchart.chart import Chart
from mplchart.primitives import Candlesticks, Volume
from mplchart.indicators import ROC, SMA, EMA, RSI, MACD
import streamlit as st
from PIL import Image
import io 
import base64



def chart(ticker):

    prices = yf.Ticker(ticker).history(period="1y", interval="1d")

    indicators = [
        Candlesticks(colordn='red',colorup='green'),SMA(10),SMA(20), SMA(50), SMA(200), Volume(),
        RSI(),
        MACD(),
    ]

    chart = Chart(title=ticker)
    chart.plot(prices, indicators)
    x = chart.render(format='png')
    image = Image.open(io.BytesIO(x))
    #st.image(image, caption=None, width=800, use_container_width=None, clamp=False, channels="RGB", output_format="PNG")buffered = io.BytesIO()
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # Save the image to a buffer in PNG format
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Display the image in Streamlit using st.markdown
    st.markdown(f"![Alt Text](data:image/png;base64,{img_base64})",unsafe_allow_html=True)



    