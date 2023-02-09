from plotly import colors
import streamlit as st
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.signal import find_peaks
import streamlit_vertical_slider as svs
import librosa
import librosa.display
import itertools
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import altair as alt
import time


# Fourier transform
def fourier(signal, sr):
    magnitude = np.fft.rfft(signal)
    frequency = np.fft.rfftfreq(len(signal), 1/sr)
    phase = np.angle(magnitude)
    return frequency, magnitude


# Inverse Fourier transform
def inverse_fourier(time, yf):
    final_signal = np.fft.irfft(yf)
    #final_signal = np.meshgrid(time)
    #plot(time, np.real(final_signal))
    return final_signal


def appendsignal(x, y):
    for time, magnitude in zip(x, y):
        st.session_state.SignalTime.append(time)
        st.session_state.SignalMagnitude.append(magnitude)


def appendfourier(x, y):
    for x_fourier, y_fourier in zip(x, y):
        st.session_state.FourierFrequency.append(x_fourier)
        st.session_state.FourierLoudness.append(y_fourier)

# Plotting Function


def plot(x, y, x_axis, y_axis, title):
    Fig = plt.figure(figsize=(6, 5))
    plt.plot(x, y)
   # plt.title(title)  #, fontdict = font1
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.plotly_chart(Fig)
    plt.show()


def getFMax(xAxis, yAxis):
    amplitude = np.abs(sc.fft.rfft(yAxis))
    frequency = sc.fft.rfftfreq(len(xAxis), (xAxis[1]-xAxis[0]))
    indices = find_peaks(amplitude)
    if len(indices[0]) > 0:
        max_freq = round(frequency[indices[0][-1]])
    else:
        max_freq = 1
    return max_freq


def generate_sliders(slidersNum, label):
    min_value = 0
    max_value = 0
    sliders_data = []
    boundary = int(50)
    columns = st.columns(slidersNum)
    for i in range(0, slidersNum):
        with columns[i]:
            min_value = 0
            max_value = 1
            slider = svs.vertical_slider(
                key=i, default_value=1, step=0.1, min_value=min_value, max_value=max_value)
            st.write(label[i])

            if slider == None:
                slider = 1
            sliders_data.append(slider)

    return sliders_data


def modifiy_vowels_signal(magnitude_freq, freqency, sliders_value):
    for i in range(len(sliders_value)):
        if sliders_value[i] == None:
            sliders_value[i] = 1

    counter = 0
    for value in freqency:
        # sh
        if value > 1895 and value < 7805:
            magnitude_freq[counter] *= sliders_value[0]
        # R
        if value > 1500 and value < 3000 or value > 500 and value < 2000:
            magnitude_freq[counter] *= sliders_value[1]
        # A
        # if value > 330 and value < 3300:
        #     magnitude_freq[counter] *= sliders_value[2]

        # range1 = value > 100 and value < 1400
        # range2 = value > 2000 and value < 6000
        # # L
        # if range1 or range2:
        #     magnitude_freq[counter] *= sliders_value[4]

        # O
        # if value > 500 and value < 2000:
        #     magnitude_freq[counter] *= sliders_value[4]
        # # Y
        # if value > 490 and value < 2800:
        #     magnitude_freq[counter] *= sliders_value[3]
        counter += 1
    return magnitude_freq


def uniform_freq(sliders, Frequency, ModulatedFourier, FourierLoudness):
    count = 0
    Modulated = ModulatedFourier
    max = 95
    if Frequency != [] and ModulatedFourier != [] and FourierLoudness != []:
        for i in range(0, 10):
            for freq in Frequency:
                if i == 0:
                    if max*(i/10) <= freq <= max*((i+1)/10):
                        Modulated[count] = FourierLoudness[count]*sliders[i]
                        count += 1
                else:
                    if max*(i/10) < freq <= max*((i+1)/10):
                        Modulated[count] = FourierLoudness[count]*sliders[i]
                        count += 1
    return Modulated


def altair_plot(original_df, Inverse_df):
    lines = alt.Chart(original_df).mark_line().encode(
        x=alt.X('0:T', axis=alt.Axis(title='Time')),
        y=alt.Y('1:Q', axis=alt.Axis(title='Amplitude'))
    ).properties(
        width=400,
        height=300
    )
    inverse_lines = alt.Chart(Inverse_df).mark_line().encode(
        x=alt.X('0:T', axis=alt.Axis(title='Time')),
        y=alt.Y('1:Q', axis=alt.Axis(title='Amplitude'))
    ).properties(
        width=400,
        height=300
    ).interactive()
    return lines


def animation(df):
    lines = alt.Chart(df).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='Time')),
        y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude')),
    ).properties(
        width=400,
        height=300
    ).interactive()

    return lines


def dynamic_plot(line_plot, original_df, Inverse_df):
    N = original_df.shape[0]
    burst = 6
    size = burst
    for i in range(1, N):
        step_df = original_df.iloc[0:size]
        inv_step_df = Inverse_df.iloc[0:size]
        lines = animation(step_df)
        inv_lines = animation(inv_step_df)
        concat = alt.hconcat(lines, inv_lines)
        line_plot = line_plot.altair_chart(concat)
        size = i + burst
        if size >= N:
            size = N - 1
        time.sleep(.00000000001)


def plot_spectrogram(csv, Y, sr, hop_length, y_axis="linear"):

    fig = plt.figure(figsize=(10, 5))  # ,facecolor='#0E1117'
    if csv == True:
        plt.specgram(Y, sr, cmap="rainbow")  # 200
        plt.tick_params(axis="x")  # , colors="#FAFAFA"
        plt.tick_params(axis="y")  # , colors="#FAFAFA"
        # , color='#FAFAFA'  Spectrogram Using matplotlib.pyplot.specgram() Method
        plt.title('Spectrogram', fontsize=6)
        plt.xlabel("Time (SEC)", fontsize=6)  # , color='#FAFAFA'
        plt.ylabel("Frequency (HZ)", fontsize=6)  # , color='#FAFAFA'
        st.pyplot(fig)
    else:
        librosa.display.specshow(Y,
                                 sr=sr,
                                 hop_length=hop_length,
                                 x_axis="time",
                                 y_axis=y_axis)
        plt.colorbar(format="%+2.f")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(clear_figure=False)
