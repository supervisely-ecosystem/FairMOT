import errno
import os
import requests
from pathlib import Path

import sly_globals as g
import supervisely_lib as sly
from sly_train_progress import get_progress_cb, reset_progress, init_progress

local_weights_path = None


def get_models_list():

    res = [
        {
            "model": "dla_34",
            "params": "20.349",
            # "flops": "7.63",
            # "top1": "68.75",
            # "top5": "88.87"
        },
        {
            "model": "resdcn_34",
            "params": "25.054",
            # "flops": "7.63",
            # "top1": "68.75",
            # "top5": "88.87"
        },
        {
            "model": "resdcn_50",
            "params": "31.190",
            # "flops": "7.63",
            # "top1": "68.75",
            # "top5": "88.87"
        },
        {
            "model": "resfpndcn_34",
            "params": "26.823",
            # "flops": "7.63",
            # "top1": "68.75",
            # "top5": "88.87"
        },

    ]
    # _validate_models_configs(res)
    return res


def get_table_columns():
    return [
        {"key": "model", "title": "Model", "subtitle": None},
        {"key": "params", "title": "Params (M)", "subtitle": None},
    ]


def get_model_info_by_name(name):
    models = get_models_list()
    for info in models:
        if info["model"] == name:
            return info
    raise KeyError(f"Model {name} not found")


def get_pretrained_weights_by_name(name):
    return get_model_info_by_name(name)["weightsUrl"]


def _validate_models_configs(models):
    res = []
    for model in models:
        model_config_path = os.path.join(g.root_source_dir, model["modelConfig"])
        train_config_path = os.path.join(g.root_source_dir, model["config"])
        if not sly.fs.file_exists(model_config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_config_path)
        if not sly.fs.file_exists(train_config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), train_config_path)
        res.append(model)
    return res


def init(data, state):
    models = get_models_list()
    data["models"] = models
    data["modelColumns"] = get_table_columns()
    state["selectedModel"] = "dla_34"  # "ResNet-50"
    state["weightsInitialization"] = "imagenet"  # "custom"  # "imagenet" #@TODO: for debug

    state["collapsed3"] = not True
    state["disabled3"] = not True
    state["modelLoading"] = False
    init_progress(3, data)

    state["weightsPath"] = ""# "/mmclassification/5687_synthetic products v2_003/checkpoints/epoch_10.pth"  #@TODO: for debug
    data["done3"] = False


def restart(data, state):
    data["done3"] = False
    # state["collapsed6"] = True
    # state["disabled6"] = True


@g.my_app.callback("apply_model")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def apply_model(api: sly.Api, task_id, context, state, app_logger):
    global local_weights_path
    try:
        if state["weightsInitialization"] == "custom":
            weights_path_remote = state["weightsPath"]
            if not weights_path_remote.endswith(".pth"):
                raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                                 f"Supported: '.pth'")

            # local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
            local_weights_path = os.path.join(g.my_app.data_dir, 'custom_model.pth')
            # api.file.download(g.team_id, weights_path_remote, local_weights_path)
            if sly.fs.file_exists(local_weights_path):
                os.remove(local_weights_path)

            file_info = g.api.file.get_info_by_path(g.team_id, weights_path_remote)
            if file_info is None:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path_remote)
            progress_cb = get_progress_cb(3, "Download weights", file_info.sizeb, is_size=True, min_report_percent=1)
            g.api.file.download(g.team_id, weights_path_remote, local_weights_path, g.my_app.cache, progress_cb)
            reset_progress(3)
        else:
            pass
        #     weights_url = get_pretrained_weights_by_name(state["selectedModel"])
        #     local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
        #     if sly.fs.file_exists(local_weights_path) is False:
        #         response = requests.head(weights_url, allow_redirects=True)
        #         sizeb = int(response.headers.get('content-length', 0))
        #         progress_cb = get_progress_cb(6, "Download weights", sizeb, is_size=True, min_report_percent=1)
        #         sly.fs.download(weights_url, local_weights_path, g.my_app.cache, progress_cb)
        #         reset_progress(6)
        # sly.logger.info("Pretrained weights has been successfully downloaded",
        #                 extra={"weights": local_weights_path})
    except Exception as e:
        reset_progress(6)
        fields = [
            {"field": "state.modelLoading", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)
        raise e

    fields = [
        {"field": "data.done3", "payload": True},
        {"field": "state.collapsed4", "payload": False},
        {"field": "state.disabled4", "payload": False},
        {"field": "state.activeStep", "payload": 4},
        {"field": "state.modelLoading", "payload": False},
    ]
    api.app.set_field(task_id, "data.scrollIntoView", f"step{4}")
    g.api.app.set_fields(g.task_id, fields)
