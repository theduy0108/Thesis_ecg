"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!

https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/
"""
import wfdb
from wfdb import processing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import BaselineWanderRemoval
from scipy.signal import butter, lfilter
from model.model import *
import tensorflow as tf
from scipy.signal import find_peaks
from datatoclound import *


def bsw_remove(signal, s_rate):
    signal = np.array(BaselineWanderRemoval.fix_baseline_wander(signal, s_rate))
    return signal


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def create_file(file_name, fs_target):

    record = wfdb.rdrecord(file_name)

    re_record = processing.resample_sig(record.p_signal[:, 0], record.fs, fs_target)

    hcut = 1
    lcut = 40
    ecg_data_bp = butter_bandpass_filter(re_record[0], hcut, lcut, fs_target, order=5)
    ecg_data_bp = bsw_remove(ecg_data_bp, fs_target)
    return ecg_data_bp


def automation_realtime_ecg(ecg_data, id):
    heart_rate_list = []
    anim_running = True
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 4.096), ylim=(-3, 3))
    ax.minorticks_on()

    # Make the major grid
    ax.grid(which='major', linestyle='-', color='red', linewidth='0.75')
    # Make the minor grid
    ax.grid(which='minor', linestyle=':', color='black', linewidth='0.5')

    line, = ax.plot([], [], lw=2, c='black')
    peaks = ax.scatter([], [], lw=2, c='red')
    hr = ax.text(1.85, 2, [], weight='bold', fontsize=20, color="red",
                 bbox={'facecolor': 'white',
                       'alpha': 1, 'pad': 15})

    stop_button = ax.text(.015, .98, [], weight='bold', fontsize=10, color="red", ha='left', va='top',
                          transform=ax.transAxes,
                          bbox={'facecolor': 'red',
                                'alpha': 1,
                                'pad': 0.5,
                                'edgecolor': 'red',
                                'boxstyle': 'round'})

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        peaks.set_offsets(np.stack(([], []), axis=1))
        hr.set_text([])
        stop_button.set_text([])
        return line, peaks, hr, stop_button

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
            database_hr = json.dumps({'Hr': heart_rate_list},
                                     cls=NumpyEncoder)
            datatocloud = data_tofrom_clound(firebaseConfig, id, database_hr)
            datatocloud.pushdata()

    def animate(i):
        x = np.linspace(0, 4.096, 125*4+12)
        y = np.array(ecg_data[i*125*4+12: i*125*4+12 + 4*125+12])
        ecg_peak, distance = find_peaks(y, height=0.2, distance=50)
        try:
            heart_rate = int(np.mean(60/np.diff(x[ecg_peak])))
            heart_rate_list.append(heart_rate)
            print('Hear rate is {}'.format(heart_rate))
        except:
            heart_rate = '_'
            print('error')

        # label = np.expand_dims(y, axis=0)
        # label = model.predict(label)
        # label = np.argmax(label, axis=-1)
        # value, ecg_peak = np.where(label > 0)

        line.set_data(x, y)
        peaks.set_offsets(np.stack((x[ecg_peak], y[ecg_peak]), axis=1))

        hr.set_text("\u2764\ufe0f {}".format(heart_rate))
        stop_button.set_text("a")
        return line, peaks, hr, stop_button

    fig.canvas.mpl_connect('button_press_event', onClick)
    func_animation = animation.FuncAnimation(fig, animate, init_func=init, frames=50, interval=300, blit=True, repeat_delay=1000)
    anim = func_animation
    plt.show()


if __name__=="__main__":

    data_path = 'D:/Thesis/Data/physionet.org/files/mitdb/1.0.0/205'
    checkpoint_dir = 'D:/Thesis/Data/model_weight/'

    ecg_data = create_file(data_path, 125)

    # model = semantic_unet_1d(
    #     input_shape=(512, 1),
    #     num_classes=7)
    # print(model.summary())
    #
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    #
    # model.compile(
    #     optimizer=optimizer, loss=loss, metrics=[jacard_coef],
    # )
    #
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    # print(model.summary)

    firebaseConfig = {
       "apiKey": "AIzaSyA88dqSHpD9oYwavanZw8U0SDcnKjGXgQM",
        "authDomain": "heartrate-measurement-124ee.firebaseapp.com",
        "databaseURL": "https://heartrate-measurement-124ee-default-rtdb.asia-southeast1.firebasedatabase.app",
        "projectId": "heartrate-measurement-124ee",
        "storageBucket": "heartrate-measurement-124ee.appspot.com",
        "messagingSenderId": "546691474661",
        "appId": "1:546691474661:web:0cc000c3a501cfd5bbe00c",
        "measurementId": "G-SYRV31M773"
    }
    id = input("Please input the gmail")
    automation_realtime_ecg(ecg_data, id)