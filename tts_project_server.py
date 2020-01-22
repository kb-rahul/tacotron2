import streamlit as st
import numpy as np
import pandas as pd
import sys
from tts_server_utils import load_taco2, load_waveglow, gen_e2e_taco


st.title("Nicole's TTS Deployed for testing")
taco_path = sys.argv[1] # path to model checkpoint 
wave_path = sys.argv[2] # path to waveglow

@st.cache(allow_output_mutation=True)
def load_model(taco_path, wave_path):
    taco_model = load_taco2(taco_path)
    waveglow, denoiser = load_waveglow(wave_path)
    return taco_model, waveglow, denoiser

data_load_state = st.text("Loading models ....!")
taco_model, wave_model, denoiser = load_model(taco_path, wave_path)
data_load_state.text("Loading models ...... Done!!")
preselect = st.radio("Select or type in the box below!", ("Thank you for contacting us", " I can surely solve your issue today", "I can talk to you in english"))
text_box = st.text_input("Enter Text here ", value="")
if not text_box:
    audio_data = gen_e2e_taco(preselect, taco_model, wave_model, denoiser)
    st.audio(audio_data, format='audio/wav', start_time=0)
else:
    audio_data = gen_e2e_taco(text_box, taco_model, wave_model, denoiser)
    st.audio(audio_data, format='audio/wav', start_time=0)
