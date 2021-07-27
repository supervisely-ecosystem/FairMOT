import os
import functools
from functools import lru_cache

import serve_globals as g
import nn_utils
import supervisely_lib as sly


@lru_cache(maxsize=10)
def get_image_by_id(image_id):
    img = g.api.image.download_np(image_id)
    return img


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.my_app.callback("get_model_meta")
@sly.timeit
@send_error_data
def get_model_meta(api: sly.Api, task_id, context, state, app_logger):
    """
    return: model meta
    """
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    """
    return: dict with info about model
    """
    model_data = nn_utils.get_model_data()

    info = {
        "app": "FairMOT Serve",
        "weights": g.remote_weights_path,
        "device": g.device,
        "session_id": task_id,
        "model arch": model_data['arch'],
        "model epoch": model_data['epoch'],
        "heads": model_data['heads'],
        "head_conv": model_data['head_conv']
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


@g.my_app.callback("inference_video_id")  # to video, input_video_id[frame_range] -> output_annotation_for_video
@sly.timeit
@send_error_data
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    video_id = state["video_id"]
    frames_range = state["frames_range"]

    nn_utils.process_video(video_id, frames_range)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=predictions)


@g.my_app.callback("inference_batch_ids")
@sly.timeit
@send_error_data
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    raise NotImplementedError("Please contact tech support")


def debug_inference():
    image_id = 903277
    image_path = f"./data/images/{image_id}.jpg"
    if not sly.fs.file_exists(image_path):
        g.my_app.public_api.image.download_path(image_id, image_path)
    res = nn_utils.inference_model(g.model, image_path, topn=5)


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    nn_utils.download_model_and_configs()
    nn_utils.construct_model_meta()
    nn_utils.deploy_model()

    g.my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
