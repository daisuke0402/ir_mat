import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import streamlit as st
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data()
def get_functional_groups(wavenumber):
    prompt = f"What are the possible functional groups that peak at this wavenumber {wavenumber} in FTIR? Dont answer in sentences, answer with words only, .If there are multiple possibilities, please answer all. "
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

@st.cache_data()
def get_japanese_translation(text):
    prompt = f"Translate the following English text to Japanese: {text}"
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

def read_csv_data(file, start_row, end_row):
    st.info(f"Reading CSV file {file.name}, please wait...")
    df = pd.read_csv(file, encoding='Shift-jis', skiprows=range(0, start_row - 1), nrows=end_row - start_row + 1, header=None)
    df.columns = ['wavenumber(cm-1)', 'Abs']
    return df

def plot_absorption_data(data):
    fig, ax = plt.subplots()
    ax.plot(data['wavenumber(cm-1)'], data['Abs'], label='Absorption')
    ax.set_xlim(4000, 400)
    ax.set_ylim(-0.05, 0.7)
    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel('Abs')
    return fig

def detect_peaks(data, threshold):
    with st.spinner(text="Detecting peaks, please wait..."):
        peaks, _ = find_peaks(data['Abs'], height=threshold)
        peak_data = data.iloc[peaks]
        peak_data['functional group'] = peak_data['wavenumber(cm-1)'].apply(get_functional_groups)
        peak_data['functional group (Japanese)'] = peak_data['functional group'].apply(get_japanese_translation)
    return peak_data

def plot_peaks_on_absorption_data(data, peak_data):
    fig, ax = plt.subplots()
    ax.plot(data['wavenumber(cm-1)'], data['Abs'], label='Absorption')
    ax.scatter(peak_data['wavenumber(cm-1)'], peak_data['Abs'], color='red', label='Peaks')
    ax.set_xlim(4000, 400)
    ax.set_ylim(-0.05, 0.7)
    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel('Abs')
    ax.legend()
    return fig

st.title('CSV Peak Detection')

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.sidebar.write(file_details)
    start_row = st.sidebar.number_input("Start row (1-indexed)", min_value=1, value=20)
    end_row = st.sidebar.number_input("End row (1-indexed)", min_value=start_row, value=3755)
    threshold = st.sidebar.number_input("Peak height threshold", min_value=0.0, value=0.07)
    
    data = read_csv_data(uploaded_file, start_row, end_row)
    fig = plot_absorption_data(data)
    st.pyplot(fig)
    
    peak_data = detect_peaks(data, threshold)
    fig_peak = plot_peaks_on_absorption_data(data, peak_data)
    st.pyplot(fig_peak)
    
    st.write("Detected Peaks")
    st.write(peak_data.style.format({'wavenumber(cm-1)': '{:.0f}', 'Abs': '{:.2f}'}))
else:
    st.warning("Please upload a CSV file to proceed.")
