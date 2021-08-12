import sys
import os

import serve_globals as g
import cv2
import ffmpeg


def init_script_arguments():
    sys.argv = []

    sys.argv.extend([f'task', 'mot'])

    device = '-1' if g.device == 'cpu' else g.device.split(':')[-1]

    sys.argv.extend([f'--gpus', f'{device}'])
    # sys.argv.extend([f'--conf_thres', '0'])
    sys.argv.extend([f'--output_format', ' text'])

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for

    rotateCode = None
    try:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    except:
        pass

    return rotateCode


def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)


def videos_to_frames(video_path, frames_range=None):
    video_name = (video_path.split('/')[-1]).split('.mp4')[0]
    output_path = os.path.join(g.input_converted, f'{video_name}')

    os.makedirs(output_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    rotateCode = check_rotation(video_path)

    while success:
        if frames_range:
            if frames_range[0] <= count <= frames_range[1]:
                if rotateCode is not None:
                    image = correct_rotation(image, rotateCode)
                cv2.imwrite(f"{output_path}/frame{count:06d}.jpg", image)  # save frame as JPEG file
        else:
            if rotateCode is not None:
                image = correct_rotation(image, rotateCode)
            cv2.imwrite(f"{output_path}/frame{count:06d}.jpg", image)  # save frame as JPEG file

        success, image = vidcap.read()
        count += 1

    fps = vidcap.get(cv2.CAP_PROP_FPS)

    g.video_data = {'index': 0, 'path': output_path,
                    'fps': fps, 'origin_path': video_path}
    return 0
