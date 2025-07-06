import streamlit as st 
import pandas as pd
import numpy as np
import requests
import torch 
import os



os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# UI configurations
st.set_page_config(page_title="ambideXtrous",
                   page_icon=":bridge_at_night:",
                   layout="centered")

                   
from sidebar import SideBar
from navigate import navigator



torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
# torch.classes.__path__ = []

SideBar()

navigator()


