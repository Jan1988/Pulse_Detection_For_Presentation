import os
import cv2
import numpy as np

from matplotlib import pyplot as plt

from POS_Based_Method import rgb_into_pulse_signal, get_bpm
from Load_Video import load_video


face_cascade_path = os.path.join('C:/', 'Anaconda3', 'pkgs', 'opencv3-3.1.0-py35_0', 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join('C:/', 'Anaconda3', 'pkgs', 'opencv3-3.1.0-py35_0', 'Library', 'etc', 'haarcascades', 'haarcascade_eye.xml')

window_size = 48

red_mean_lst = []
green_mean_lst = []
blue_mean_lst = []
channel_means_lst = []

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# vom webcam pulse programm
face_rect = [1, 1, 2, 2]
last_center = np.array([0, 0])
find_faces = True
buffer_size = window_size * 3
col = (100, 255, 100)

roi_means_buffer = []
h_signal_buffer = []



def draw_rect(rect, frame_out):
    x, y, w, h = rect
    cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return frame_out


def calc_shift(detected, last_center):
    x, y, w, h = detected
    center = np.array([x + 0.5 * w, y + 0.5 * h])
    shift = np.linalg.norm(center - last_center)

    last_center = center
    return shift, last_center


def get_subface_coord(fh_x, fh_y, fh_w, fh_h):
    x, y, w, h = face_rect

    return [int(x + w * fh_x - (w * fh_w / 2.0)),
            int(y + h * fh_y - (h * fh_h / 2.0)),
            int(w * fh_w),
            int(h * fh_h)]


def get_subface_means(coord, frame_in):
    x, y, w, h = coord
    subframe = frame_in[y:y + h, x:x + w, :]

    channel_means = np.mean(subframe, axis=(0, 1))

    return channel_means


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # print('frame: ' + str(j))
    frame_clone = frame.copy()

    gray_frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    detected = list(face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50),
                                                  flags=cv2.CASCADE_SCALE_IMAGE))

    if len(detected) > 0:
        detected.sort(key=lambda a: a[-1] * a[-2])

        shift, last_center = calc_shift(detected[-1], last_center)
        if shift > 8:
            face_rect = detected[-1]

    forehead1 = get_subface_coord(0.5, 0.18, 0.25, 0.15)

    frame_clone = draw_rect(face_rect, frame_clone)

    x, y, w, h = face_rect
    cv2.putText(frame_clone, "Face", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
    frame_clone = draw_rect(forehead1, frame_clone)
    x, y, w, h = forehead1
    cv2.putText(frame_clone, "Forehead", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)

    channel_means = get_subface_means(forehead1, frame)
    roi_means_buffer.append(channel_means)

    L = len(roi_means_buffer)
    if L > buffer_size:
        roi_means_buffer = roi_means_buffer[-buffer_size:]
        L = buffer_size

    if L > window_size:

        h_signal, norm_rgb = rgb_into_pulse_signal(roi_means_buffer)
        # h_signal_buffer.append(h_signal)

        bpm, bandpassed_fft, heart_rates, fft, raw = get_bpm(h_signal, 25)
        cv2.putText(frame_clone, "Current BPM: %s" % str(bpm), (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25,
                    col)

    small = cv2.resize(frame_clone, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('frame_clone', small)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

fig = plt.figure(figsize=(15, 10))
fig.suptitle('BPM: ' + str(bpm), fontsize=20, fontweight='bold')


sub1 = fig.add_subplot(311)
plt.plot(norm_rgb[:, 2], 'r', norm_rgb[:, 1], 'g', norm_rgb[:, 0], 'b')

sub2 = fig.add_subplot(312)
plt.plot(h_signal, 'k')

sub3 = fig.add_subplot(313)
plt.plot(heart_rates, bandpassed_fft, 'k')

plt.show()








