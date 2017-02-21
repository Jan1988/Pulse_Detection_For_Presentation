import os
import cv2
import math
import numpy as np

from matplotlib import pyplot as plt
from POS_Based_Method import extract_pos_based_method_improved
from Load_Video import load_video


# Rounds up to multiple
def roundUp(numToRound, multiple):

    if multiple == 0:
        return numToRound

    remainder = abs(numToRound) % multiple

    if remainder == 0:
        return numToRound

    if numToRound < 0:
        return -(abs(numToRound) - remainder)

    return int(numToRound + multiple - remainder)


def outputFrame(frame, center, w, h):
    x = int(center[0] - w / 2)
    y = int(center[1] - 3 * h / 5)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame

def detect_faces(frame_with_faces, _prior_center):



    # Convert to gray scale and equalize histogram
    # gray_frame = cv2.equalizeHist(cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2GRAY))
    gray_frame = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.4, 2, minSize=(50, 50))

    for (x, y, w, h) in faces:


        face_center = (int(x + w/2), int(y + h/2))

        # Generate width and height of face, round to closest 1 / 4 of frame height
        h = roundUp(h, frame_with_faces.shape[0] / 4)
        w = int(3 * h / 5)

        if _prior_center == 0:
            _prior_center = face_center
            break

        # Check to see if it's probably the same user
        if abs(face_center[0] - prior_center[0]) < frame_with_faces.shape[1] and abs(face_center[1] - prior_center[1]) < frame_with_faces.shape[0]:

            # Check to see if the user moved enough to update position
            if abs(face_center[0] - _prior_center[0]) < 10 and abs(face_center[1] - _prior_center[1]) < 10:
                face_center = _prior_center

            # Smooth new center compared to old center
            face_center_new = (int((face_center[0] + 2 * _prior_center[0]) / 3), int((face_center[1] + 2 * _prior_center[1]) / 3))



            #
            x = int(face_center[0] - w / 2)
            y = int(face_center[1] - 3 * h / 5)
            #
            print(x, y, w, h)
            cv2.rectangle(frame_with_faces, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_color = frame_with_faces[y:y + h, x:x + w]
            #
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 2, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            #
            if len(eyes) < 1:
                face_center_new = _prior_center
                break




            cv2.circle(frame_with_faces, face_center, 3, (0, 0, 0), -1)
            cv2.circle(frame_with_faces, _prior_center, 3, (0, 255, 0), -1)




            _prior_center = face_center_new
            # exit if primary users face probably found
            break

    return frame_with_faces, _prior_center


if __name__ == '__main__':

    face_cascade_path = os.path.join('C:/', 'Anaconda3', 'pkgs', 'opencv3-3.1.0-py35_0', 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join('C:/', 'Anaconda3', 'pkgs', 'opencv3-3.1.0-py35_0', 'Library', 'etc', 'haarcascades', 'haarcascade_eye.xml')
    dir_path = os.path.join('assets')
    file = '00080.MTS'
    file_path = os.path.join(dir_path, file)

    window_numbers = 6
    window_size = 48
    frame_count = window_numbers * window_size + 1

    video_frames, fps = load_video(file_path)
    video_frames = video_frames[1:frame_count]
    print('Reduced Frame Count: ' + str(len(video_frames)))

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # Create time series array of the roi means
    viola_roi_sequence = []
    prior_center = (0, 0)

    for j, frame in enumerate(video_frames):
        frame_clone = frame.copy()

            # x2 = int(x * 1.15)
            # y2 = int(y * 1.1)
            # w2 = int(w * 0.25)
            # h2 = int(h * 0.175)
            #
            # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # roi_gray = gray[y:y + h, x:x + w]
            # roi_color = frame[y:y + h, x:x + w]
            #
            # frame = cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 0), 2)
            # roi_frame = frame_clone[y2:y2 + h2, x2:x2 + w2]

        frame, prior_center = detect_faces(frame, prior_center)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # viola_roi_sequence.append(roi_face)
        # blurred_roi = cv2.blur(roi_face, (5, 5))

    #
    # bpm, fft, heart_rates, raw_fft, H, pulse_signal, green_avg = extract_pos_based_method_improved(viola_roi_sequence, fps)
    # plot_results(green_avg,  pulse_signal, H, raw_fft, fft, heart_rates)
