import errno
import os
import requests
from pathlib import Path

import torch

import sly_globals as g
import supervisely_lib as sly
from sly_train_progress import get_progress_cb, reset_progress, init_progress
from sly_train_progress import _update_progress_ui

from functools import partial

local_weights_path = None


def init(data, state):
    state["collapsed2"] = True
    state["disabled2"] = True
    state["modelLoading"] = False
    init_progress(2, data)

    state["weightsPath"] = ""  # "/mmclassification/5687_synthetic products v2_002/checkpoints/epoch_10.pth"  #@TODO: for debug
    data["done2"] = False


def restart(data, state):
    data["done2"] = False
    # state["collapsed6"] = True
    # state["disabled6"] = True
    sly.fs.clean_dir(g.checkpoints_dir)


def list_files(sly_fs_path):
    nesting_level = len(sly_fs_path.split('/'))

    files_in_dir = g.api.file.list(g.team_id, sly_fs_path)
    pth_paths = [file['path'] for file in files_in_dir if file['path'].endswith('.pth') and
                 len(file['path'].split('/')) == nesting_level]

    return pth_paths


def get_file_sizes(sly_fs_path):
    size_b = []

    files = g.api.file.list(g.team_id, sly_fs_path)
    for file in files:
        size_b.append(file['meta']['size'])

    return sum(size_b)


def download_checkpoints(sly_fs_path):
    if sly.fs.dir_exists(g.checkpoints_dir):
        sly.fs.clean_dir(g.checkpoints_dir)

    files_size_b = get_file_sizes(sly_fs_path)
    download_progress = get_progress_cb(2, "Download checkpoints", files_size_b, is_size=True, min_report_percent=1)

    g.api.file.download_directory(g.team_id, sly_fs_path, g.checkpoints_dir, progress_cb=download_progress)

    reset_progress(2)
    return 0


def generate_rows_by_models(models_paths):
    rows = []

    for model_path in models_paths:
        try:
            curr_model = torch.load(model_path, map_location=torch.device('cpu'))
            arch = curr_model['arch']
            epoch = curr_model['epoch']
            del curr_model
        except Exception as ex:
            print(ex)
            arch = epoch = '-'
        rows.append({
            "name": f"{model_path.split('/')[-1]}",
            "arch": f"{arch}",
            "epoch": f"{epoch}",
            "isDisabled": False if arch != '-' else True,
        })

    return rows


def fill_table(table_rows):
    fields = [
        {"field": f"data.modelsTable", "payload": table_rows, "recursive": False},
    ]
    g.api.task.set_fields(g.task_id, fields)

    return 0


@g.my_app.callback("load_models_handler")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def load_models_handler(api: sly.Api, task_id, context, state, app_logger):
    try:
        pth_paths = list_files(state["weightsPath"])
        if len(pth_paths) == 0:
            raise FileNotFoundError('Not found .pth files in directory')
        download_checkpoints(state["weightsPath"])

        models_paths_local = [os.path.join(g.checkpoints_dir, model_name) for model_name in
                              os.listdir(g.checkpoints_dir) if model_name.endswith('.pth')]
        table_rows = generate_rows_by_models(models_paths_local)

        # for row_name in rows_names:
        #     if not row_name['isDisabled']:
        #         state['selectedClass'] = row_name['name']
        #         break
        #
        table_rows = sorted(table_rows, key=lambda k: k['isDisabled'])

        fill_table(table_rows)

    except Exception as e:
        reset_progress(2)
        fields = [
            {"field": "state.modelLoading", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)
        raise e

    fields = [
        {"field": "data.done2", "payload": True},
        {"field": "state.collapsed3", "payload": False},
        {"field": "state.disabled3", "payload": False},
        {"field": "state.activeStep", "payload": 3},
        {"field": "state.modelLoading", "payload": False},
    ]
    # api.app.set_field(task_id, "data.scrollIntoView", f"step{3}")
    g.api.app.set_fields(g.task_id, fields)
