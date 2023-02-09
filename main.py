import io
from logging import raiseExceptions
import os
import os.path
import librosa.display
import plotly.graph_objects as go
import plotly.express as px
from scipy.io import wavfile
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from PIL import Image
import numpy as np
import streamlit as st
import librosa
import streamlit_vertical_slider as svs
import pandas as pd
import functions as functions
import sys
import time as tm
from scipy.io.wavfile import write
import copy


# Initializing lists for time and frequency domain
if 'SignalTime' not in st.session_state:
    st.session_state.SignalTime = []
if 'SignalMagnitude' not in st.session_state:
    st.session_state.SignalMagnitude = []
if 'FourierFrequency' not in st.session_state:
    st.session_state.FourierFrequency = []
if 'FourierLoudness' not in st.session_state:
    st.session_state.FourierLoudness = []
if 'ModulatedFourierLoudness' not in st.session_state:
    st.session_state.ModulatedFourierLoudness = []


# Setting the layout of the page
img = Image.open('clipart-music ong.png')
st.set_page_config(page_title="Signal Equalizer", page_icon=img, layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)


st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)


hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: ;}

            
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")


def empty_lists():
    st.session_state.SignalTime = []
    st.session_state.SignalMagnitude = []
    st.session_state.FourierFrequency = []
    st.session_state.FourierLoudness = []
    st.session_state.ModulatedFourierLoudness = []


# Browsing section
with st.sidebar:
    uploaded_file = st.file_uploader(
        'Upload file', type=["wav", "csv"], accept_multiple_files=False)
    options = st.radio("Select an Option", ('Sinusoidals',
                       'Music instrumentations', 'Vowels', 'Animals'))
    convert_btn = st.button('Apply')
    display_options = st.selectbox(
        'Display Options', ('Static display', 'Dynamic display', 'Spectogram'))


if uploaded_file is not None:

    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1][1:]
    data = []

    if uploaded_file.type == "audio/wav":
        empty_lists()
        data, samplerate = librosa.load(
            uploaded_file, sr=None, mono=True, offset=0.0, duration=None)
        sample_frequency = 1/samplerate
        fmax = sample_frequency/2
        # the length of the generated sample
        duration = len(data)/samplerate
        timeWav = np.arange(0, duration, 1/samplerate)
        st.sidebar.audio(file_name)

        # spectrogram calculations
        scale, sr = librosa.load(file_name)
        FRAME_SIZE = 2048
        HOP_SIZE = 512
        S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        Y_scale = np.abs(S_scale) ** 2
        Y_log_scale = librosa.power_to_db(Y_scale)

    if ext == 'csv':
        empty_lists()
        df = pd.read_csv(uploaded_file)
        list_of_columns = df.columns
        time = df[list_of_columns[0]].to_numpy()
        data = df[list_of_columns[1]].to_numpy()
        max_freq = functions.getFMax(time, data)
        samplerate = 2*max_freq
        duration = len(time)/samplerate
        HOP_SIZE = 512

    if options == 'Sinusoidals':
        if not ext == 'csv':
            st.warning('Please Upload a CSV file')
            sys.exit('Please Upload a CSV file')

        label = ['0-9.5 HZ', '9.5-19 HZ', '19-28.5 HZ', '28.5-38 HZ', '38-47.5 HZ',
                 '47.5-57 HZ', '57-66.5 HZ', '66.5-76 HZ', '76-85.5 HZ', '85.5-95 HZ']
        if st.session_state.SignalTime == [] or st.session_state.SignalMagnitude == []:
            for time, magnitude in zip(df["time"], df["values"]):
                st.session_state.SignalTime.append(time)
                st.session_state.SignalMagnitude.append(magnitude)

        if st.session_state.FourierFrequency == [] or st.session_state.FourierLoudness == []:
            freq, yf = functions.fourier(
                st.session_state.SignalMagnitude, samplerate)
            for x_fourier, y_fourier in zip(freq, yf):
                st.session_state.FourierFrequency.append(x_fourier)
                st.session_state.FourierLoudness.append(y_fourier)

        if st.session_state.ModulatedFourierLoudness == []:
            for amplitude in st.session_state.FourierLoudness:
                st.session_state.ModulatedFourierLoudness.append(amplitude)

        sliders_data1 = functions.generate_sliders(10, label)

        st.session_state.ModulatedFourierLoudness = functions.uniform_freq(sliders_data1, st.session_state.FourierFrequency,
                                                                           st.session_state.ModulatedFourierLoudness, st.session_state.FourierLoudness)

        inverse_sig = functions.inverse_fourier(
            st.session_state.SignalTime, st.session_state.ModulatedFourierLoudness)
        if len(st.session_state.SignalTime) % 2 != 0:
            del st.session_state.SignalTime[len(st.session_state.SignalTime)-1]

        if display_options == 'Static display':
            #functions.plot(st.session_state.FourierFrequency,st.session_state.FourierLoudness,"Time(Sec)", "Value (volt)", "Original Signal")
            col1, col2 = st.columns(2)
            with col1:
                functions.plot(df["time"], df["values"],
                               "Time(Sec)", "Value (volt)", "Original Signal")

            with col2:
                functions.plot(st.session_state.SignalTime, np.real(
                    inverse_sig), "Time(Sec)", "Value (volt)", "Applied Signal")

        elif display_options == 'Dynamic display':
            original_df = pd.DataFrame(
                {'time': st.session_state.SignalTime[::20], 'amplitude': st.session_state.SignalMagnitude[::20]}, columns=['time', 'amplitude'])
            Inverse_df = pd.DataFrame(
                {'time': st.session_state.SignalTime[::20], 'amplitude': inverse_sig[::20]}, columns=['time', 'amplitude'])
            lines = functions.altair_plot(original_df, Inverse_df)
            line_plot = st.altair_chart(lines)
            functions.dynamic_plot(line_plot, original_df, Inverse_df)

        elif display_options == 'Spectogram':
            col1, col2 = st.columns(2)
            with col1:
                functions.plot_spectrogram(
                    True, st.session_state.SignalMagnitude, samplerate, HOP_SIZE, y_axis="log")
            with col2:
                abs_new_value = np.abs(inverse_sig)
                functions.plot_spectrogram(
                    True, abs_new_value, samplerate, HOP_SIZE, y_axis="log")

    if options == 'Music instrumentations':
        if not ext == 'wav':
            st.warning('Please Upload an audio wav file')
            sys.exit('Please Upload an audio wav file')

        if not data == []:

            if st.session_state.SignalTime == [] or st.session_state.SignalMagnitude == []:
                for time, magnitude in zip(timeWav, data):
                    st.session_state.SignalTime.append(time)
                    st.session_state.SignalMagnitude.append(magnitude)

            st.session_state.FourierFrequency, st.session_state.FourierLoudness = functions.fourier(
                st.session_state.SignalMagnitude, samplerate)

            st.session_state.ModulatedFourierLoudness = st.session_state.FourierLoudness.copy()

            music_label = ['Drums', 'Trumphet', 'xylophone']
            sliders_data2 = functions.generate_sliders(
                slidersNum=3, label=music_label)

            count1 = 0
            for i in range(0, len(st.session_state.FourierFrequency)):
                if 0 <= st.session_state.FourierFrequency[i] <= 280:
                    st.session_state.ModulatedFourierLoudness[
                        i] = st.session_state.ModulatedFourierLoudness[i]*sliders_data2[0]
                    # 130-250 flamenco guitar range a3-g3  "0-250"

                elif 280 < st.session_state.FourierFrequency[i] <= 1000:

                    # if 250 < freq <= 600:
                    st.session_state.ModulatedFourierLoudness[
                        i] = st.session_state.ModulatedFourierLoudness[i]*sliders_data2[1]

                    # 250-500 classical piano range a4-g4  "250-600"

                elif 1000 < st.session_state.FourierFrequency[i] <= 7000:

                    # if 600< freq <=20000:
                    st.session_state.ModulatedFourierLoudness[
                        i] = st.session_state.ModulatedFourierLoudness[i]*sliders_data2[2]

            inverse_sig = functions.inverse_fourier(
                st.session_state.SignalTime, st.session_state.ModulatedFourierLoudness)

            if convert_btn:
                norm = np.int16((inverse_sig)*(32767/inverse_sig.max()))
                write('Edited_audio.wav', round(samplerate), norm)
                st.sidebar.audio('Edited_audio.wav', format='audio/wav')

            if display_options == 'Static display':
                col1, col2 = st.columns(2)
                with col1:
                    functions.plot(st.session_state.SignalTime,
                                   st.session_state.SignalMagnitude, 'Time (sec)', 'Magnitude', 'Original')

                with col2:
                    if len(st.session_state.SignalTime) % 2 != 0:
                        del st.session_state.SignalTime[len(
                            st.session_state.SignalTime)-1]
                    functions.plot(st.session_state.SignalTime, np.real(
                        inverse_sig), 'Freq (Hz)', 'Amplitude', 'Applied')

            elif display_options == 'Dynamic display':
                original_df = pd.DataFrame(
                    {'time': st.session_state.SignalTime[::20], 'amplitude': st.session_state.SignalMagnitude[::20]}, columns=['time', 'amplitude'])

                Inverse_df = pd.DataFrame(
                    {'time': st.session_state.SignalTime[::20], 'amplitude': inverse_sig[::20]}, columns=['time', 'amplitude'])
                lines = functions.altair_plot(original_df, Inverse_df)
                line_plot = st.altair_chart(lines)
                functions.dynamic_plot(line_plot, original_df, Inverse_df)

            elif display_options == 'Spectogram':
                col1, col2 = st.columns(2)
                with col1:
                    functions.plot_spectrogram(
                        False, Y_log_scale, sr, HOP_SIZE, y_axis="log")

    if options == 'Vowels':

        if not data == []:

            # vowels_labels = ["sh", "R", "A"]
            vowels_labels = ["sh", "R"]

            slider = functions.generate_sliders(2, vowels_labels)

            # ---------------------------------------------------------------------

            if st.session_state.SignalTime == [] or st.session_state.SignalMagnitude == []:
                for time, magnitude in zip(timeWav, data):
                    st.session_state.SignalTime.append(time)
                    st.session_state.SignalMagnitude.append(magnitude)

            st.session_state.FourierFrequency, st.session_state.FourierLoudness = functions.fourier(
                data, samplerate)

            # if st.session_state.ModulatedFourierLoudness == []:
            #     for amplitude in st.session_state.FourierLoudness:
            #         st.session_state.ModulatedFourierLoudness.append(amplitude)

            st.session_state.ModulatedFourierLoudness = functions.modifiy_vowels_signal(
                st.session_state.FourierLoudness, st.session_state.FourierFrequency, slider)

            inverse_sig = functions.inverse_fourier(
                st.session_state.SignalTime, st.session_state.ModulatedFourierLoudness)

            # bytes_wav = bytes()
            # byte_io = io.BytesIO(bytes_wav)
            # write(byte_io, samplerate, inverse_sig.astype(np.float32))
            # result_bytes = byte_io.read()
            # st.sidebar.audio(result_bytes, format="audio/wav")

            norm = np.int16((inverse_sig)*(32767/inverse_sig.max()))
            write('Edited_audio.wav', round(samplerate), norm)
            st.sidebar.audio('Edited_audio.wav', format='audio/wav')

            # st.write("output.wav", inverse_sig, samplerate)
            # st.sidebar.audio("output.wav")

            if display_options == 'Static display':
                if len(st.session_state.SignalTime) > len(st.session_state.SignalMagnitude):
                    if len(st.session_state.SignalTime) % 2 != 0:
                        del st.session_state.SignalTime[len(
                            st.session_state.SignalTime)-1]

                if len(st.session_state.SignalMagnitude) > len(st.session_state.SignalTime):
                    if len(st.session_state.SignalMagnitude) % 2 != 0:
                        del st.session_state.SignalMagnitude[len(
                            st.session_state.SignalMagnitude)-1]

                col1, col2 = st.columns(2)
                with col1:
                    functions.plot(st.session_state.SignalTime,
                                   st.session_state.SignalMagnitude, 'Time', 'Magnitude', 'Original')

                with col2:
                    inverse_abs = np.real(inverse_sig)
                    functions.plot(st.session_state.SignalTime,
                                   inverse_abs, 'Time', 'Magnitude', 'Applied')

            elif display_options == 'Dynamic display':
                original_df = pd.DataFrame(
                    {'time': st.session_state.SignalTime[::20], 'amplitude': st.session_state.SignalMagnitude[::20]}, columns=['time', 'amplitude'])

                Inverse_df = pd.DataFrame(
                    {'time': st.session_state.SignalTime[::20], 'amplitude': inverse_sig[::20]}, columns=['time', 'amplitude'])

                # Inverse_df = pd.DataFrame(
                #     {'time': st.session_state.FourierFrequency[::20], 'amplitude': np.abs(st.session_state.FourierLoudness[::20])}, columns=['time', 'amplitude'])
                lines = functions.altair_plot(original_df, Inverse_df)
                line_plot = st.altair_chart(lines)
                functions.dynamic_plot(line_plot, original_df, Inverse_df)

            elif display_options == 'Spectogram':
                col1, col2 = st.columns(2)
                with col1:
                    functions.plot_spectrogram(
                        False, Y_log_scale, sr, HOP_SIZE, y_axis="log")
                # with col2 :
                #     inverse_sig =functions.inverse_fourier(st.session_state.SignalTime,st.session_state.ModulatedFourierLoudness)
                #     Y_inv = np.abs(inverse_sig) ** 2
                #     Y_log_inv = librosa.power_to_db(Y_inv)
                #     functions.plot_spectrogram(False,inverse_sig, sr, HOP_SIZE, y_axis="log")

    if options == 'Animals':
        if not ext == 'wav':
            st.warning('Please Upload an audio wav file')
            sys.exit('Please Upload an audio wav file')

        if not data == []:

            if st.session_state.SignalTime == [] or st.session_state.SignalMagnitude == []:
                for time, magnitude in zip(timeWav, data):
                    st.session_state.SignalTime.append(time)
                    st.session_state.SignalMagnitude.append(magnitude)

            st.session_state.FourierFrequency, st.session_state.FourierLoudness = functions.fourier(
                st.session_state.SignalMagnitude, samplerate)

            if st.session_state.ModulatedFourierLoudness == []:
                for amplitude in st.session_state.FourierLoudness:
                    st.session_state.ModulatedFourierLoudness.append(amplitude)

            animal_label = ['Lion', 'Eagle']
            sliders_data2 = functions.generate_sliders(
                slidersNum=2, label=animal_label)

            count = 0
            for freq in st.session_state.FourierFrequency:

                if 0 <= freq <= 2000:
                    st.session_state.ModulatedFourierLoudness[
                        count] = st.session_state.FourierLoudness[count]*sliders_data2[0]

                if 2000 < freq <= 7000:
                    st.session_state.ModulatedFourierLoudness[
                        count] = st.session_state.FourierLoudness[count]*sliders_data2[1]

                # if 7000<freq<=4000:
                #         st.session_state.ModulatedFourierLoudness[count]= st.session_state.FourierLoudness[count]*sliders_data2[2]

                count += 1

            # if len(st.session_state.ModulatedFourierLoudness) % 2 != 0:
            #     del st.session_state.ModulatedFourierLoudness[len(
            #         st.session_state.ModulatedFourierLoudness)-1]

            inverse_sig = functions.inverse_fourier(
                st.session_state.SignalTime, st.session_state.ModulatedFourierLoudness)

            if display_options == 'Static display':
                col1, col2 = st.columns(2)
                with col1:
                    functions.plot(st.session_state.SignalTime,
                                   st.session_state.SignalMagnitude, 'Time (sec)', 'Magnitude', 'Original')

                with col2:
                    if len(st.session_state.SignalTime) % 2 != 0:
                        del st.session_state.SignalTime[len(
                            st.session_state.SignalTime)-1]
                    functions.plot(st.session_state.SignalTime, np.real(
                        inverse_sig), 'Time (sec)', 'Magnitude', 'Original')

            elif display_options == 'Dynamic display':
                original_df = pd.DataFrame(
                    {'time': st.session_state.SignalTime[::20], 'amplitude': st.session_state.SignalMagnitude[::20]}, columns=['time', 'amplitude'])

                Inverse_df = pd.DataFrame(
                    {'time': st.session_state.SignalTime[::20], 'amplitude': inverse_sig[::20]}, columns=['time', 'amplitude'])
                lines = functions.altair_plot(original_df, Inverse_df)
                line_plot = st.altair_chart(lines)
                functions.dynamic_plot(line_plot, original_df, Inverse_df)

            elif display_options == 'Spectogram':
                col1, col2 = st.columns(2)
                with col1:
                    functions.plot_spectrogram(
                        False, Y_log_scale, sr, HOP_SIZE, y_axis="log")

            if convert_btn:
                norm = np.int16((inverse_sig)*(32767/inverse_sig.max()))
                write('Edited_audio.wav', round(samplerate), norm)
                st.sidebar.audio('Edited_audio.wav', format='audio/wav')

                # with col2 :
                #     inverse_sig =functions.inverse_fourier(st.session_state.SignalTime,st.session_state.ModulatedFourierLoudness)
                #     Y_inv = np.abs(inverse_sig) ** 2
                #     Y_log_inv = librosa.power_to_db(Y_inv)
                #     functions.plot_spectrogram(False,inverse_sig, sr, HOP_SIZE, y_axis="log")


else:
    empty_lists()
