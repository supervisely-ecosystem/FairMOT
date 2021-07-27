import json
import os
import supervisely_lib as sly
from sly_train_progress import get_progress_cb, reset_progress, init_progress
import sly_globals as g

from sly_train_progress import _update_progress_ui
from sly_train_args import init_script_arguments
from functools import partial

from lib.opts import opts
from train import main as fair_mot_train
from gen_data_path import gen_data_path
from gen_labels import gen_labels

import shutil

from supervisely_lib.app.widgets import CompareGallery
import sly_train_renderer

import glob

_open_lnk_name = "open_app.lnk"


def init(data, state):
    init_progress("TrainInfo", data)
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)

    data["etaEpoch"] = None
    data["etaIter"] = None
    data["etaEpochData"] = []

    state["trainStarted"] = False
    state["finishTrain"] = False
    state["finishTrainDialog"] = False

    init_charts(data, state)

    gallery_custom = CompareGallery(g.task_id, g.api, f"data.galleryPreview", g.project_meta)
    data[f"galleryPreview"] = gallery_custom.to_json()

    data["previewPredLinks"] = []
    state["currEpochPreview"] = 1

    state["collapsed6"] = not True
    state["disabled6"] = not True
    state["done6"] = False

    data["outputName"] = None
    data["outputUrl"] = None


def restart(data, state):
    data["done6"] = False


def init_chart(title, names, xs, ys, smoothing=None, yrange=None, decimals=None, xdecimals=None):
    series = []
    for name, x, y in zip(names, xs, ys):
        series.append({
            "name": name,
            "data": [[px, py] for px, py in zip(x, y)]
        })
    result = {
        "options": {
            "title": title,
            # "groupKey": "my-synced-charts",
        },
        "series": series
    }
    if smoothing is not None:
        result["options"]["smoothingWeight"] = smoothing
    if yrange is not None:
        result["options"]["yaxisInterval"] = yrange
    if decimals is not None:
        result["options"]["decimalsInFloat"] = decimals
    if xdecimals is not None:
        result["options"]["xaxisDecimalsInFloat"] = xdecimals
    return result


def init_charts(data, state):
    # demo_x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    # demo_y = [[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]

    data["chartLoss"] = init_chart("Loss", names=["train"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartHmLoss"] = init_chart("HM Loss", names=["hm"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartWhLoss"] = init_chart("WH Loss", names=["wh"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartOffLoss"] = init_chart("OFF Loss", names=["offset"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartIdLoss"] = init_chart("ID Loss", names=["id"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)

    data["chartValPrecision"] = init_chart("Precision", names=["precision"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartValRecall"] = init_chart("Recall", names=["recall"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartValMap"] = init_chart("mAP", names=["mAP"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)


    #
    # data["chartTime"] = init_chart("Time", names=["time"], xs=[[]], ys=[[]], xdecimals=2)
    # data["chartDataTime"] = init_chart("Data Time", names=["data_time"], xs=[[]], ys=[[]], xdecimals=2)
    # data["chartMemory"] = init_chart("Memory", names=["memory"], xs=[[]], ys=[[]], xdecimals=2)
    state["smoothing"] = 0.6


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_train_results():
    _save_link_to_ui(g.experiment_dir, g.my_app.app_url)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.Progress):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)

    progress = sly.Progress("Upload directory with training files to Team Files", 0, is_size=True)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)

    exp_id = g.api.app.get_field(g.task_id, 'state.expId')
    remote_dir = f"/FairMOT/train/{exp_id}"
    res_dir = g.api.file.upload_directory(g.team_id, g.experiment_dir, remote_dir, progress_size_cb=progress_cb)
    return res_dir



def dump_config(root_path):
    mot_config = {
        "root": root_path,
        "train":
            {
                "sly_mot": f"./data/sly_mot.train"
            },
        "test":
            {
                "sly_mot": f"./data/sly_mot.test"
            }
    }

    with open(f'{g.my_app.data_dir}/sly_mot_generated.json', 'w') as file:
        # mot_config = json.dumps(mot_config)
        json.dump(mot_config, file)


def remove_negative_labels(root_path):
    class_label = g.api.app.get_field(g.task_id, 'state.selectedClass')

    gt_files = g.get_files_paths(root_path, [f'.txt'])

    for gt_file in gt_files:
        if not gt_file.endswith(f'{class_label}.txt'):
            os.remove(gt_file)


def organize_data(state):

    root_path = f'{g.my_app.data_dir}/data/SLY_MOT'

    if os.path.exists(root_path):  # clear SLY_MOT data dirs before start organizing
        shutil.rmtree(root_path)

    train_videos_paths = state['trainVideosPaths']
    val_videos_paths = state['valVideosPaths']

    organize_in_mot_format(video_paths=train_videos_paths, is_train=True)
    organize_in_mot_format(video_paths=val_videos_paths, is_train=False)

    remove_negative_labels(root_path)

    gen_data_path(root_path)
    gen_labels(root_path)

    clean_exp_dir()

    dump_config(root_path)


def clean_exp_dir():
    exp_id = g.api.app.get_field(g.task_id, 'state.expId')
    exp_dir = f"../exp/mot/{exp_id}/"

    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)


def get_class_info():
    table = g.api.app.get_field(g.task_id, 'data.selectClassTable')
    selected_class_label = g.api.app.get_field(g.task_id, 'state.selectedClass')
    return [row for row in table if row['name'] == selected_class_label][0]


def dump_logs(state):
    exp_id = g.api.app.get_field(g.task_id, 'state.expId')
    checkpoints_dir = f"../exp/mot/{exp_id}/"

    info_files_paths = g.get_files_paths(checkpoints_dir, ['.txt'])
    for info_files_path in info_files_paths:
        destination = os.path.join(g.logs_dir, info_files_path.split('/')[-1])
        shutil.move(info_files_path, destination)


def dump_checkpoints():
    exp_id = g.api.app.get_field(g.task_id, 'state.expId')

    checkpoints_dir = f"../exp/mot/{exp_id}/"

    checkpoints_paths_src = g.get_files_paths(checkpoints_dir, ['.pth'])

    for checkpoints_path in checkpoints_paths_src:
        destination = os.path.join(g.checkpoints_dir, checkpoints_path.split('/')[-1])
        shutil.move(checkpoints_path, destination)


def dump_info(state):
    preview_pred_links = g.api.app.get_field(g.task_id, 'data.previewPredLinks')

    sly.json.dump_json_file(get_class_info(), os.path.join(g.info_dir, "class_info.json"))
    sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))
    sly.json.dump_json_file(preview_pred_links, os.path.join(g.info_dir, "preview_pred_links.json"))


def dump_results(state):
    dump_logs(state)
    dump_checkpoints()
    dump_info(state)


def organize_in_mot_format(video_paths=None, is_train=True):
    organize_progress = get_progress_cb("TrainInfo", f"Organizing {'train' if is_train else 'validation'} data",
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

    reset_progress("TrainInfo")


@g.my_app.callback("setFinishTrainFlag")
@sly.timeit
def set_finish_train_flag(api: sly.Api, task_id, context, state, app_logger):
    g.api.app.set_fields(g.task_id,
                         [{'field': 'state.finishTrain', 'payload': True}])


@g.my_app.callback("previewByEpoch")
@sly.timeit
def preview_by_epoch(api: sly.Api, task_id, context, state, app_logger):
    if len(g.api.app.get_field(g.task_id, 'data.previewPredLinks')) > 0:
        index = int(state['currEpochPreview'] / state["valInterval"]) - 1

        gallery_preview = CompareGallery(g.task_id, g.api, f"data.galleryPreview", g.project_meta)
        sly_train_renderer.update_preview_by_index(index, gallery_preview)


@g.my_app.callback("train")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        sly_dir_path = os.getcwd()
        os.chdir('../../../src')

        organize_data(state)
        init_script_arguments(state)

        opt = opts().parse()

        fair_mot_train(opt)

        dump_results(state)
        # hide progress bars and eta
        fields = [
            {"field": "data.progressEpoch", "payload": None},
            {"field": "data.progressIter", "payload": None},
            {"field": "data.etaEpoch", "payload": None},
            {"field": "data.etaIter", "payload": None},
        ]
        g.api.app.set_fields(g.task_id, fields)

        remote_dir = upload_train_results()
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        # show result directory in UI
        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.done6", "payload": True},
            {"field": "state.finishTrain", "payload": False},
            {"field": "state.trainStarted", "payload": False},
        ]
        # sly_train_renderer.send_fields({'data.eta': None})  # SLY CODE
        g.api.app.set_fields(g.task_id, fields)
    except Exception as e:
        api.app.set_field(task_id, "state.trainStarted", False)
        raise e  # app will handle this error and show modal window

    os.chdir(sly_dir_path)
    # stop application
    # g.my_app.stop()
