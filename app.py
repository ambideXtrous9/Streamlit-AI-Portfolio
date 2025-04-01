import streamlit as st 
import pandas as pd
import numpy as np
import requests
import torch 
import os

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
# torch.classes.__path__ = []

from sidebar import SideBar
from navigate import navigator

# UI configurations
st.set_page_config(page_title="ambideXtrous",
                   page_icon=":bridge_at_night:",
                   layout="centered")

SideBar()

navigator()


