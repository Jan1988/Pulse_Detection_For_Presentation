import os
import cv2

from Load_Video import load_video
from Viola_Smooth import findFaceGetPulse

if __name__ == '__main__':


    dir_path = os.path.join('assets')
    file = '00080.mkv'
    file_path = os.path.join(dir_path, file)

    processor = findFaceGetPulse(bpm_limits=[50, 160], data_spike_limit=2500., face_detector_smoothness=10.)

    video_frames, fps = load_video(file_path)

    for j, frame in enumerate(video_frames):

        # set current image frame to the processor's input
        processor.frame_in = frame

        # process the image frame to perform all needed analysis
        processor.run(0)
        # collect the output frame for display
        output_frame = processor.frame_out

        # show the processed/annotated output frame
        cv2.imshow("Processed", output_frame)

