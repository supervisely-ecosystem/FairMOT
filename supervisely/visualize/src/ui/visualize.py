import json
import os
import supervisely_lib as sly
from sly_train_progress import get_progress_cb, reset_progress, init_progress
import sly_globals as g

from sly_train_progress import _update_progress_ui

from functools import partial

from lib.opts import opts
from sly_track import main as track
from gen_data_path import gen_data_path
from gen_labels import gen_labels

import sys
import torch
import re
from PIL import Image

import shutil
import cv2


import glob

_open_lnk_name = "open_app.lnk"


def init(data, state):
    init_progress("Models", data)
    init_progress("Videos", data)
    init_progress("UploadDir", data)

    data["etaEpoch"] = None
    data["etaIter"] = None
    data["etaEpochData"] = []
    data['gridPreview'] = None

    state["visualizingStarted"] = False

    state["collapsed5"] = not True
    state["disabled5"] = not True
    state["done5"] = False

    data["outputName"] = None
    data["outputUrl"] = None


def restart(data, state):
    data["done5"] = False


def init_script_arguments(state):
    sys.argv = []

    sys.argv.extend([f'task', 'mot'])

    def camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    needed_args = ['expId', 'gpus', 'confThres']

    for needed_arg in needed_args:
        sys.argv.extend([f'--{camel_to_snake(needed_arg)}', f'{state[needed_arg]}'])

    sys.argv.extend([f'--data_cfg', f'{g.my_app.data_dir}/sly_mot_generated.json'])
    sys.argv.extend([f'--output-format', ' video'])
    sys.argv.extend([f'--output-root', '../demo_output/'])


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_visualization_results():

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.Progress):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)

    progress = sly.Progress("Upload directory with visualization files to Team Files", 0, is_size=True)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)

    exp_id = g.api.app.get_field(g.task_id, 'state.expId')
    remote_dir = f"/FairMOT/visualization/{exp_id}"
    local_dir = os.path.join(g.output_dir, exp_id)

    _save_link_to_ui(local_dir, g.my_app.app_url)

    res_dir = g.api.file.upload_directory(g.team_id, local_dir, remote_dir, progress_size_cb=progress_cb)

    return res_dir




#
#
#
# def dump_info(state):
#     preview_pred_links = g.api.app.get_field(g.task_id, 'data.previewPredLinks')
#
#     sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))
#     sly.json.dump_json_file(preview_pred_links,
#                             os.path.join(g.info, "preview_pred_links.json"))

def organize_in_mot_format(video_paths=None, is_train=True):
    organize_progress = get_progress_cb("VisualizeInfo", f"Organizing {'train' if is_train else 'validation'} data",
                                        (len(video_paths)))

    working_dir = 'train' if is_train else 'test'

    mot_general_dir = os.path.join(g.my_app.data_dir, 'data', 'SLY_MOT')
    mot_images_path = os.path.join(mot_general_dir, f'images/{working_dir}')
    os.makedirs(mot_images_path, exist_ok=True)

    for video_index, video_path in enumerate(video_paths):
        split_path = video_path.split('/')
        project_id = split_path[-3]
        ds_name = split_path[-2]

        destination = os.path.join(mot_images_path, f'{project_id}_{ds_name}_{video_index}')

        if not os.path.exists(destination):  # DEBUG
            shutil.copytree(video_path, destination)

        organize_progress(1)

    reset_progress("VisualizeInfo")


def get_visualizing_checkpoints(max_count):
    models_rows = g.api.app.get_field(g.task_id, 'data.modelsTable')
    selected_models = g.api.app.get_field(g.task_id, 'state.selectedModels')

    visualizing_models = []
    for row in models_rows:
        if row['name'] in selected_models:
            visualizing_models.append(row)

    visualizing_models = sorted(visualizing_models, key=lambda k: int(k['epoch']))

    if len(visualizing_models) <= max_count:
        return visualizing_models
    else:
        step = int(len(visualizing_models) / max_count)
        return [visualizing_models[index] for index in range(0, len(visualizing_models), step)][:max_count]


def get_visualization_video_paths(visualization_models):
    exp_id = g.api.app.get_field(g.task_id, 'state.expId')
    res_path = os.path.join(g.output_dir, exp_id)
    checkpoints_names = os.listdir(os.path.join(g.output_dir, exp_id))

    video_paths = []

    for visualization_model in visualization_models:
        folder_name = visualization_model['name'].replace('.', '_')
        if folder_name in checkpoints_names:
            videos_path = os.path.join(res_path, folder_name, 'videos')
            video_name = sorted(os.listdir(videos_path))[0]
            video_paths.append(os.path.join(videos_path, video_name))

    return video_paths


def get_video_shape(video_path):
    vcap = cv2.VideoCapture(video_path)
    height = width = 0
    if vcap.isOpened():
        width = vcap.get(3)  # float `width`
        height = vcap.get(4)

    return tuple([int(width), int(height)])


def check_video_shape(video_paths):
    video_shape = (0, 0)
    for index, video_path in enumerate(video_paths):
        if index == 0:
            video_shape = get_video_shape(video_path)
        else:
            if get_video_shape(video_path) != video_shape:
                return None
    return video_shape


def create_image_placeholder(video_shape):
    img = Image.new('RGB', video_shape, color='black')
    image_placeholder_path = os.path.join(g.grid_video_dir, 'placeholder.png')
    img.save(image_placeholder_path)


def generate_row(row_paths):
    created_videos = glob.glob(f'{g.grid_video_dir}/*.mp4')
    video_num = len(created_videos)
    output_video_path = f'{g.grid_video_dir}/{video_num}.mp4'

    while len(row_paths) % 3 != 0:  # filling empty grid space by placeholder
        row_paths.append(f'{g.grid_video_dir}/placeholder.png')

    input_args = ' -i '.join(row_paths)

    cmd_str = f'ffmpeg -y -i {input_args} -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" ' \
              f'-map "[v]" -c:v libx264 {output_video_path}'
    os.system(cmd_str)


def generate_grid():
    exp_id = g.api.app.get_field(g.task_id, 'state.expId')
    created_videos = sorted(glob.glob(f'{g.grid_video_dir}/*.mp4'))
    output_video_path = os.path.join(f'{g.output_dir}', f'{exp_id}', 'grid_preview.mp4')

    if len(created_videos) == 1:
        shutil.copy(created_videos[0], output_video_path)
    else:
        input_args = ' -i '.join(created_videos)

        cmd_str = f'ffmpeg -y -i {input_args} -filter_complex vstack={len(created_videos)} -c:v libx264 {output_video_path}'
        os.system(cmd_str)


def generate_grid_video(video_paths):
    video_shape = check_video_shape(video_paths)
    if video_shape:
        create_image_placeholder(video_shape)
        for index in range(0, len(video_paths), 3):
            row_paths = video_paths[index: index + 3]
            generate_row(row_paths)
        generate_grid()
        return 0
    else:
        return -1






@g.my_app.callback("visualize_videos")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def visualize_videos(api: sly.Api, task_id, context, state, app_logger):
    sly_dir_path = os.getcwd()
    os.chdir('../../../src')
    try:

        init_script_arguments(state)

        opt = opts().init()
        # opt = opt.parse()

        track(opt)

        checkpoints_list = get_visualizing_checkpoints(max_count=9)
        video_paths = get_visualization_video_paths(checkpoints_list)
        generate_grid_video(video_paths)


        # hide progress bars and eta
        fields = [
            {"field": "data.progressModels", "payload": None},
            {"field": "data.progressVideos", "payload": None},
        ]
        g.api.app.set_fields(g.task_id, fields)

        remote_dir = upload_visualization_results()
        grid_file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, 'grid_preview.mp4'))
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))

        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        # show result directory in UI
        fields = [
            {"field": "data.gridPreview", "payload": grid_file_info.full_storage_url},
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.done5", "payload": True},
            {"field": "state.visualizingStarted", "payload": False},
        ]
        # sly_train_renderer.send_fields({'data.eta': None})  # SLY CODE
        g.api.app.set_fields(g.task_id, fields)
    except Exception as e:
        api.app.set_field(task_id, "state.visualizingStarted", False)
        raise e  # app will handle this error and show modal window

    os.chdir(sly_dir_path)
    # stop application
    # g.my_app.stop()
