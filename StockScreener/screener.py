import yfinance as yf
from ta.trend import sma_indicator
import streamlit as st
import requests
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from gnews import GNews
from langchain_core.messages import AIMessage
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import yfinance as yf
import ta
import os 
import re

load_dotenv()




nifty500_df = pd.read_csv("StockScreener/ind_nifty500list.csv")
microcap250_df = pd.read_csv("StockScreener/ind_niftymicrocap250_list.csv") 

nifty500_df['YFSYMBOL'] = nifty500_df['Symbol'] + '.NS'
microcap250_df['YFSYMBOL'] = microcap250_df['Symbol'] + '.NS'

df500 = list(nifty500_df['YFSYMBOL'])
microcap250 = list(microcap250_df['YFSYMBOL'])

complist = list(nifty500_df['Company Name'])

def get_yf_symbol(company_name: str):
    match = nifty500_df.loc[nifty500_df['Company Name'] == company_name, 'YFSYMBOL']
    return match.iloc[0] if not match.empty else None

rocket_icon = '<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="50" height="50" />'
heading = f"## {rocket_icon} AI Stock Financial Research Report"

chart_icon = '<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Chart%20Increasing%20with%20Yen.png" alt="Chart Increasing with Yen" width="40" height="40" />'
final_report = f"### {chart_icon} Final Report"

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


llm = ChatGroq(
    model_name="qwen/qwen3-32b",
    temperature=0.7
)


# Initialize the GNews object
google_news = GNews(language='en', period='30d',max_results=20)

from StockScreener.mlpchart.mlpchart import chart

from langgraph.prebuilt import create_react_agent

stock_agent = create_react_agent(
        model=llm,
        tools=[],
        prompt = (
            """
            Role:
            You are an Expert Equity Research Analyst with 20+ years of experience. Your task is to rigorously analyze a stock using multi-dimensional data and provide an institutional-grade, data-driven recommendation.

            Methodology:
            Follow the **ReAct Framework**. For every key step, label it as:
            - Thought: Explain your reasoning
            - Action: Specify what data/metrics you are analyzing
            - Observation: State what the data reveals with factual insights

            Data Inputs to Analyze:
            • **Fundamentals**: P/E, P/B, ROE, Debt/Equity, YoY and QoQ revenue & profit growth  
            • **Technicals**: RSI, EMA(10/20), SMA(50/100/200), MACD, Volume, Support/Resistance  
            • **Sentiment & News**: Recent developments, macro or sector trends, government policies  
            • **Ownership Patterns**: FII/DII trends, promoter holding changes (last 2 quarters), pledging  
            • **Valuation Models**: Compare current valuation with intrinsic value (DCF, Comparables) and sector average  
            • **Risk-Reward**: Margin of Safety (MOS), Upside/Downside potential

            Output Structure:
            1. *Recommendation*
            - **Call**: 'Buy', 'Sell', or 'Hold' with 
            - **Conviction Level**: (High/Medium/Low)
            - Justify with Valuation, Technicals, Financial trends, Institutional/Promoter activity, Sector outlook
            - Mention **Current Price** and **% Upside/Downside**

            2. *Entry Analysis*
            - Is this a good entry point? Backed by:
                • Valuation vs Sector & Historical Range  
                • Technical indicators (e.g. RSI, SMA200 crossover, volume spikes)  
                • FII/DII activity or Insider Buying  
                • Sector Outlook and Competitive Position  

            3. *Price Targets*
            - Provide Base, Bull, and Bear targets for:  
                • Short Term (3 months)  
                • Medium Term (6 months)  
                • Long Term (12 months)  
            - Use a mix of technical levels (e.g. Fibonacci extensions) and fundamental valuation

            4. *Risks & Catalysts*
            - Highlight Red Flags: High debt, pledging, litigation, governance, slowdown risks  
            - Highlight Tailwinds: Policy incentives, strong order book, capacity expansion, M&A, new product launches  

            5. *Final Verdict*
            - Conclude with a **3-line summary** (e.g., "Buy for 25% upside to 12M target of ₹XXX, driven by strong EPS growth, low valuation, and sector tailwinds. Entry near ₹YYY. Stop loss: ₹ZZZ.")

            Tone:
            - Professional, concise, data-rich. Use numbers wherever possible.
            - Prioritize clarity and insights over verbosity.
            - Use bullet points or short paragraphs for better readability.
            """
        )

    )

# ---------------------------
# 🧑‍🔬 Stock Researcher Agent
# ---------------------------
def stock_node(fundamentals,shareholding,technical_indicators,news):
    # Prepare the prompt
    user_msg = {
        "role": "user",
        "content": (
            f"Do the research on the Stock based on provided data and latest news:\n\n"
            f"**Fundamentals**: {fundamentals}\n\n**Yearly and Quarterly Profit/Loss Data and Shareholding of FII and DIIs**: {shareholding}\n\n**Technical Indicators**: {technical_indicators}\n\n**News**: {news}"
        )
    }


    ai_content = ""
    for step in stock_agent.stream({"messages": [user_msg]}, stream_mode="values"):
        msg = step["messages"][-1]
        if isinstance(msg, AIMessage):
            ai_content = msg.content
            
    return ai_content



def compute_latest_technical_indicators(ticker: str):
    # Fetch historical price data
    data = yf.Ticker(ticker).history(period="1y", interval="1d")
    data.dropna(inplace=True)

    # Compute Moving Averages
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_100'] = data['Close'].rolling(window=100).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Compute RSI
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()

    # Compute MACD
    macd = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()

    # Select only the latest row
    latest = data.iloc[-1]

    # Return as dictionary
    result = {
        'Close': round(latest['Close'], 2),
        'Volume': int(latest['Volume']),
        'EMA_10': round(latest['EMA_10'], 2),
        'EMA_20': round(latest['EMA_20'], 2),
        'SMA_50': round(latest['SMA_50'], 2),
        'SMA_100': round(latest['SMA_100'], 2),
        'SMA_200': round(latest['SMA_200'], 2),
        'RSI': round(latest['RSI'], 2),
        'MACD': round(latest['MACD'], 2),
        'MACD_Signal': round(latest['MACD_Signal'], 2),
        'MACD_Diff': round(latest['MACD_Diff'], 2)
    }

    return result






def BreakoutVolume(niftylist):
    stockList = []
    
    total_items = len(niftylist)
    itr = 0
    progress_bar = st.progress(0)

    # Cache for storing stock data
    stock_cache = {}
    
    # Get current date for week and month calculations
    current_date = pd.Timestamp.now()
    current_week = current_date.strftime('%Y-%U')
    current_month = current_date.to_period('M')

    for symbol in niftylist:
        progress_bar.progress((itr + 1) / total_items)
        itr += 1

        # Check cache first
        if symbol in stock_cache:
            dt = stock_cache[symbol]
        else:
            try:
                stock = yf.Ticker(symbol)
                dt = stock.history(period="6mo", interval="1d")
                if not dt.empty:
                    dt = dt.reset_index()
                    stock_cache[symbol] = dt
                else:
                    continue
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue

        # Early exit if not enough data
        if len(dt) < 5:
            continue

        # Calculate SMAs and EMA in one go
        dt['50_SMA'] = sma_indicator(dt['Close'], window=50)
        dt['20_SMA'] = sma_indicator(dt['Close'], window=20)
        dt['Volume_EMA20'] = dt['Volume'].ewm(span=20, adjust=False).mean()

        # Sort once
        dt.sort_values(by='Date', ascending=False, inplace=True)

        # Get daily values
        daily_values = dt.iloc[0]
        prev_day_values = dt.iloc[1]
        two_days_ago_values = dt.iloc[2]
        three_days_ago_values = dt.iloc[3]
        four_days_ago_values = dt.iloc[4]

        # Filter by volume and price first (most likely to fail)
        if daily_values['Volume'] < daily_values['Volume_EMA20'] or \
           daily_values['Close'] < daily_values['50_SMA'] or \
           daily_values['Close'] < daily_values['20_SMA']:
            continue

        # Get week and month data
        # Get week and month data
        # Sort by date in ascending order to get the most recent data
        dt_sorted = dt.sort_values(by='Date', ascending=True)
        
        # Get the most recent data point for week and month
        week_filtered = dt_sorted[dt_sorted['Date'].dt.strftime('%Y-%U') == current_week]
        month_filtered = dt_sorted[dt_sorted['Date'].dt.to_period('M') == current_month]
        
        if week_filtered.empty or month_filtered.empty:
            # If we don't have data for current week/month, try getting the most recent available data
            week_data = dt_sorted.iloc[-1]
            month_data = dt_sorted.iloc[-1]
            #print(f"Warning: Using most recent data for {symbol} instead of current week/month")
        else:
            week_data = week_filtered.iloc[-1]
            month_data = month_filtered.iloc[-1]

        # Calculate price ranges
        daily_range = abs(daily_values['High'] - daily_values['Low'])
        prev_range = abs(prev_day_values['High'] - prev_day_values['Low'])
        two_day_range = abs(two_days_ago_values['High'] - two_days_ago_values['Low'])
        three_day_range = abs(three_days_ago_values['High'] - three_days_ago_values['Low'])
        four_day_range = abs(four_days_ago_values['High'] - four_days_ago_values['Low'])

        # Check price ranges
        if not (daily_range > prev_range and 
               daily_range > two_day_range and 
               daily_range > three_day_range and 
               daily_range > four_day_range):
            continue

        # Check closing conditions
        if not (daily_values['Close'] > daily_values['Open'] and 
               daily_values['Close'] > week_data['Open'] and 
               daily_values['Close'] > month_data['Open']):
            continue

        # Check low condition
        if daily_values['Low'] <= (prev_day_values['Close'] - abs(prev_day_values['Close'] / 222)):
            continue

        # If we've made it this far, add to list
        stockList.append(symbol)

    return stockList

def results(soup):

    yearly_values = []
    quarter_values = []

    # Find the section with id "profit-loss"
    section = soup.find('section', id='profit-loss')

    if section:
        # Extract rows from this section
        rows = section.find_all('tr')

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'Net Profit' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                yearly_values = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row



          # Find the section with id "profit-loss"
    section = soup.find('section', id='quarters')

    if section:
        # Extract rows from this section
        rows = section.find_all('tr')

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'Net Profit' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                quarter_values = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row


    return  quarter_values, yearly_values


def shareholding(soup):

    Promoters = []
    DII = []
    FII = []
    Public = []

          # Find the section with id "profit-loss"
    section = soup.find('section', id='shareholding')

    if section:
        # Extract rows from this section
        rows = section.find_all('tr')

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'Promoters' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                Promoters = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'DIIs' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                DII = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'FIIs' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                FII = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row

        for row in rows:
            # Check if the row contains the text "Net Profit"
            if 'Public' in row.get_text():
                # Find all <td> elements in the row, skipping the first <td> which contains the button
                columns = row.find_all('td')[1:]
                Public = [col.get_text(strip=True) for col in columns]
                break  # Exit loop once we find the correct row

    return Promoters, DII, FII, Public
  
def extract_key_insights(soup):
    company_name = soup.find('h1', class_='margin-0 show-from-tablet-landscape').text.strip()
    current_price = soup.find('div', class_='font-size-18 strong line-height-14').find('span').text.strip()
    market_cap = soup.find('li', {'data-source': 'default'}).find('span', class_='number').text.strip()
    about_section = soup.find('div', class_='company-profile').find('div', class_='sub show-more-box about').text.strip()
    pe_value = soup.find('span', class_='name', string=lambda t: t and "Stock P/E" in t).find_next('span', class_='number').string
    roe = soup.find('span', class_='name', string=lambda t: t and "ROE" in t).find_next('span', class_='number').string
    roce = soup.find('span', class_='name', string=lambda t: t and "ROCE" in t).find_next('span', class_='number').string

    quarter_values, yearly_values = results(soup)
    Promoters, DII, FII, Public = shareholding(soup)

    fundainfo = {
        "Company Name": company_name,
        "Current Price": current_price,
        "Market Cap": market_cap,
        "About": about_section,
        "PE" : pe_value,
        "ROE" : roe,
        "ROCE" : roce,}

    shareholdnres = {"Quarter" : quarter_values,
        "Yearly" : yearly_values,
        "Promoters" : Promoters,
        "DII" : DII,
        "FII" : FII,
        "Public" : Public
    }

    return fundainfo, shareholdnres 


def scrapper(stock_ticker):
    
    stock_ticker = stock_ticker.replace('.NS', '')

    url = f"https://www.screener.in/company/{stock_ticker}/"

    response = requests.get(url)

    if response.status_code == 200:
        print("Successfully fetched the webpage")
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')
    
    
    fundainfo, shareholdnres = extract_key_insights(soup)
    
    return fundainfo, shareholdnres


def companyDetails(company_data):
    # Streamlit layout
    st.title(company_data['Company Name'])

    # Display current price and market cap in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Price", company_data['Current Price'])
    with col2:
        st.metric("Market Cap (₹ Cr)", company_data['Market Cap'])

    # Display PE, ROE, and ROCE in columns
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("P/E Ratio", company_data['PE'])
    with col4:
        st.metric("ROE (%)", company_data['ROE'])
    with col5:
        st.metric("ROCE (%)", company_data['ROCE'])

    # Display the "About" section with some emphasis
    st.subheader("About")
    st.markdown(f"*{company_data['About']}*")


def CompanyNews(name):
    # Fetch news articles
    news = google_news.get_news(name)

    # Display news titles with URLs in Streamlit
    st.title(f"Latest News on {name}")

    if news:
        for article in news:
            
            title = article.get('title', 'No title available')
            url = article.get('url', '#')
            
            # Display the title as a clickable link
            st.markdown(f"[{title}]({url})")
    else:
        st.write("No news found for this topic.")

    return news
    


def plotChart(symbol):
    stock = yf.Ticker(symbol)

    df = stock.history(period="1y", interval="1d")
    df = df.reset_index()

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'])])
    
    st.plotly_chart(fig, use_container_width=True)
    


# Helper function to convert percentage strings to floats
def convert_to_float(data):
    if not data or not isinstance(data, list):
        return []
    
    if all(isinstance(item, str) for item in data):
        # Remove commas and convert percentages
        if all('%' in item for item in data):
            return [float(value.replace(',', '').strip('%')) for value in data]
        # Remove commas and convert numeric strings
        else:
            return [float(value.replace(',', '')) for value in data]
    return []

def check_status(values):
        if not values or len(values) < 2:
            st.subheader(f"⚠️ :orange[Data not available]")
            return
        
        last_val = values[-1]
        second_last_val = values[-2]

        if last_val < 0:
            st.subheader(f"📉 :red[LOSS]")
        elif last_val == second_last_val:
            st.subheader(f"📈 :rainbow[Unchnaged ⚪]")
        elif last_val > second_last_val:
            st.subheader(f"📈 :green[Increased ✅]")
        else:
            st.subheader(f"📉 :red[Decreased ❌]")

def check_status_public(values):
        last_val = values[-1]
        second_last_val = values[-2]

        if last_val > second_last_val:
            st.subheader(f"📈 :red[Increased ❌]")
        else:
            st.subheader(f"📉 :green[Decreased ✅]")


def analyze_financial_data(data):
    # Function to check and return the status with appropriate coloring
    data = {key: convert_to_float(value) for key, value in data.items()}
    
    col1, col2 = st.columns(2)
    
    
    with col1:
        # Check Quarterly and Yearly data
        st.subheader("1. Quarterly Profit Status:")
        check_status(data.get('Quarter', []))

        st.subheader("2. Yearly Profit Status:")
        check_status(data.get('Yearly', []))

        # Check Shareholding data (Promoters, DII, FII, Public)
        st.subheader("3. Promoters Shareholding Status:")
        check_status(data.get('Promoters', []))
        
    with col2:
        st.subheader("4. DII Shareholding Status:")
        check_status(data.get('DII', []))

        st.subheader("5. FII Shareholding Status:")
        check_status(data.get('FII', []))

        st.subheader("6. Public Shareholding Status:")
        check_status_public(data['Public'])

def plotShareholding(shareholdnres):

    # Convert percentages in each list where necessary
    converted_data = {key: convert_to_float(value) for key, value in shareholdnres.items()}

    # Filter out empty lists
    filtered_data = {key: value for key, value in converted_data.items() if value}

    # Determine the number of subplots
    num_plots = len(filtered_data)
    rows = (num_plots + 1) // 2

    # Create subplots based on the number of non-empty data lists
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=list(filtered_data.keys())
    )

    # Plot each non-empty dataset
    for i, (key, values) in enumerate(filtered_data.items(), start=1):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        fig.add_trace(
            go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode='lines+markers',
                    name=key, line=dict(width=2)),
            row=row, col=col
        )

        fig.update_xaxes(title_text=key, row=row, col=col)
        fig.update_yaxes(title_text="Net Profit" if key in ["Quarter", "Yearly"] else "Holding (%)", row=row, col=col)

    # Update layout
    fig.update_layout(
        height=rows * 400,
        width=1000,
        title_text="Financial Data Analysis",
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def StockScan():
    """
    Main function to scan and analyze stocks
    """
    
    # Initialize selected option from session state
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None
    if 'stockList' not in st.session_state:
        st.session_state.stockList = []
    if 'nifty500_stockList' not in st.session_state:
        st.session_state.nifty500_stockList = []
    if 'microcap250_stockList' not in st.session_state:
        st.session_state.microcap250_stockList = []
    if 'analysis_stockList' not in st.session_state:
        st.session_state.analysis_stockList = []

    # Two side-by-side buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Volume Breakout NIFTY500", key='nifty500_btn'):
            st.session_state.selected_option = "Volume Breakout NIFTY500"

    with col2:
        if st.button("📈 Volume Breakout MICROCAP250", key='microcap250_btn'):
            st.session_state.selected_option = "Volume Breakout MICROCAP250"

    with col3:
        if st.button("📊 Stocks Analysis", key='analysis_btn'):
            st.session_state.selected_option = "Stocks Analysis"

    # Get the current selection from session state
    selected_option = st.session_state.selected_option

    # Show results based on selected option
    if selected_option == "Volume Breakout NIFTY500":
        st.title("Running Scan on NIFTY500")
        if st.button("Run Scan", key='nifty500_run_btn'):
            stockList = BreakoutVolume(df500)
            st.session_state.nifty500_stockList = stockList

        if st.session_state.nifty500_stockList:
            st.success(f'Scan Complete : {len(st.session_state.nifty500_stockList)} Stocks Found', icon="✅")
            st.subheader("Stocks")
            cols = st.columns(2)
            for i, stock in enumerate(st.session_state.nifty500_stockList):
                cols[i % 2].write(stock)
            
            option = st.selectbox(
                "List of Stocks",
                st.session_state.nifty500_stockList,
                index=None,
                placeholder="Select the Stock",
            )

            if option:
                fundainfo, shareholdnres = scrapper(option)
                technical_indicators = compute_latest_technical_indicators(option)
                companyDetails(fundainfo) 
                chart(ticker=option)
                plotChart(option)
                analyze_financial_data(shareholdnres)
                plotShareholding(shareholdnres)
                news = CompanyNews(fundainfo['Company Name'])

                st.markdown(heading, unsafe_allow_html=True)
                report = stock_node(fundainfo,shareholdnres,technical_indicators,news)
        
                # Extract <think> section
                think_match = re.search(r"<think>(.*?)</think>", report, re.DOTALL)
                thinking_part = think_match.group(1).strip() if think_match else "No reasoning available."
                report_without_think = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()

                with st.chat_message("StockAgent"):

                        with st.expander("🧠 Agent Reasoning (Click to Expand)"):
                            st.markdown(thinking_part)

                        st.markdown(final_report, unsafe_allow_html=True)
                        st.markdown(report_without_think)

    elif selected_option == "Volume Breakout MICROCAP250":
        st.title("Running Scan on MICROCAP250")
        if st.button("Run Scan", key='microcap250_run_btn'):
            stockList = BreakoutVolume(microcap250)
            st.session_state.microcap250_stockList = stockList

        if st.session_state.microcap250_stockList:
            st.success(f'Scan Complete : {len(st.session_state.microcap250_stockList)} Stocks Found', icon="✅")
            st.subheader("Stocks")
            cols = st.columns(2)
            for i, stock in enumerate(st.session_state.microcap250_stockList):
                cols[i % 2].write(stock)
            
            option = st.selectbox(
                "List of Stocks",
                st.session_state.microcap250_stockList,
                index=None,
                placeholder="Select the Stock",
            )

            if option:
                fundainfo, shareholdnres = scrapper(option)
                technical_indicators = compute_latest_technical_indicators(option)
                companyDetails(fundainfo) 
                chart(ticker=option)
                plotChart(option)
                analyze_financial_data(shareholdnres)
                plotShareholding(shareholdnres)
                news = CompanyNews(fundainfo['Company Name'])

                st.markdown(heading, unsafe_allow_html=True)
                report = stock_node(fundainfo,shareholdnres,technical_indicators,news)
        
                # Extract <think> section
                think_match = re.search(r"<think>(.*?)</think>", report, re.DOTALL)
                thinking_part = think_match.group(1).strip() if think_match else "No reasoning available."
                report_without_think = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()

                with st.chat_message("StockAgent"):

                        with st.expander("🧠 Agent Reasoning (Click to Expand)"):
                            st.markdown(thinking_part)

                        st.markdown(final_report, unsafe_allow_html=True)
                        st.markdown(report_without_think)

    elif selected_option == "Stocks Analysis":
        option = st.selectbox(
            "List of Stocks",
            complist,
            index=None,
            placeholder="Select the Stock",
        )

        st.write("You selected:", option)

        option = get_yf_symbol(option)
        
        if option:
            fundainfo, shareholdnres = scrapper(option)
            technical_indicators = compute_latest_technical_indicators(option)
            companyDetails(fundainfo) 
            chart(ticker=option)
            plotChart(option)
            analyze_financial_data(shareholdnres)
            plotShareholding(shareholdnres)
            news = CompanyNews(fundainfo['Company Name'])

            st.markdown(heading, unsafe_allow_html=True)
            report = stock_node(fundainfo,shareholdnres,technical_indicators,news)
    
            # Extract <think> section
            think_match = re.search(r"<think>(.*?)</think>", report, re.DOTALL)
            thinking_part = think_match.group(1).strip() if think_match else "No reasoning available."
            report_without_think = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()

            with st.chat_message("StockAgent"):

                    with st.expander("🧠 Agent Reasoning (Click to Expand)"):
                        st.markdown(thinking_part)

                    st.markdown(final_report, unsafe_allow_html=True)
                    st.markdown(report_without_think)
    
    elif selected_option == "Volume Breakout MICROCAP250":
        stockList = []
        if "stockList" not in st.session_state:
            st.session_state.stockList = []
            
        if st.button("Run Scan"):
            st.title("Running Scan on MICROCAP250")
            stockList = BreakoutVolume(microcap250)
            st.session_state.stockList = stockList  # Store in session state
            
        
        if st.session_state.stockList:
            st.success(f'Scan Complete : {len(st.session_state.stockList)} Stocks Found', icon="✅")
            
            st.subheader("Stocks")
            cols = st.columns(2)
            for i, stock in enumerate(st.session_state.stockList):
                cols[i % 2].write(stock)
            
            option = st.selectbox(
                "List of Stocks",
                st.session_state.stockList,
                index=None,
                placeholder="Select the Stock",
            )

            st.write("You selected:", option)
            
            if option:
                fundainfo, shareholdnres = scrapper(option)
                technical_indicators = compute_latest_technical_indicators(option)
                companyDetails(fundainfo) 
                chart(ticker=option)
                plotChart(option)
                analyze_financial_data(shareholdnres)
                plotShareholding(shareholdnres)
                news = CompanyNews(fundainfo['Company Name'])

                st.markdown(heading, unsafe_allow_html=True)
                report = stock_node(fundainfo,shareholdnres,technical_indicators,news)
        
                # Extract <think> section
                think_match = re.search(r"<think>(.*?)</think>", report, re.DOTALL)
                thinking_part = think_match.group(1).strip() if think_match else "No reasoning available."
                report_without_think = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()

                with st.chat_message("StockAgent"):

                        with st.expander("🧠 Agent Reasoning (Click to Expand)"):
                            st.markdown(thinking_part)

                        st.markdown(final_report, unsafe_allow_html=True)
                        st.markdown(report_without_think)
    
    elif selected_option == "Stocks Analysis":

        option = st.selectbox(
                "List of Stocks",
                complist,
                index=None,
                placeholder="Select the Stock",
            )

        st.write("You selected:", option)
        
        symbol = None
        if option:
            symbol = get_yf_symbol(option)

        
        
        if symbol:
            fundainfo, shareholdnres = scrapper(symbol)
            technical_indicators = compute_latest_technical_indicators(symbol)
            companyDetails(fundainfo) 
            chart(ticker=symbol)
            plotChart(symbol)
            analyze_financial_data(shareholdnres)
            plotShareholding(shareholdnres)
            news = CompanyNews(fundainfo['Company Name'])

            st.markdown(heading, unsafe_allow_html=True)
            report = stock_node(fundainfo,shareholdnres,technical_indicators,news)
    
            # Extract <think> section
            think_match = re.search(r"<think>(.*?)</think>", report, re.DOTALL)
            thinking_part = think_match.group(1).strip() if think_match else "No reasoning available."
            report_without_think = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL).strip()

            with st.chat_message("StockAgent"):

                    with st.expander("🧠 Agent Reasoning (Click to Expand)"):
                        st.markdown(thinking_part)

                    st.markdown(final_report, unsafe_allow_html=True)
                    st.markdown(report_without_think)
        

                
           
           
           
        
        

    
    