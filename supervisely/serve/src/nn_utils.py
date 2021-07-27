import os
import torch
import supervisely_lib as sly
# from mmcls.apis import init_model
# from mmcv.parallel import collate, scatter
# from mmcls.datasets.pipelines import Compose
from lib.models.model import create_model, load_model
from lib.opts import opts
from sly_track import eval_seq
import lib.datasets.dataset.jde as datasets

import mot_utils

import json

import serve_globals as g


@sly.timeit
def download_model_and_configs():
    if not g.remote_weights_path.endswith(".pth"):
        raise ValueError(f"Unsupported weights extension {sly.fs.get_file_ext(g.remote_weights_path)}. "
                         f"Supported extension: '.pth'")

    info = g.api.file.get_info_by_path(g.team_id, g.remote_weights_path)
    if info is None:
        raise FileNotFoundError(f"Weights file not found: {g.remote_weights_path}")

    progress = sly.Progress("Downloading weights", info.sizeb, is_size=True, need_info_log=True)
    g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(g.remote_weights_path))
    g.api.file.download(
        g.team_id,
        g.remote_weights_path,
        g.local_weights_path,
        cache=g.my_app.cache,
        progress_cb=progress.iters_done_report
    )

    def _download_dir(remote_dir, local_dir):
        remote_files = g.api.file.list2(g.team_id, remote_dir)
        progress = sly.Progress(f"Downloading {remote_dir}", len(remote_files), need_info_log=True)
        for remote_file in remote_files:
            local_file = os.path.join(local_dir, sly.fs.get_file_name_with_ext(remote_file.path))
            if sly.fs.file_exists(local_file):  # @TODO: for debug
                pass
            else:
                g.api.file.download(g.team_id, remote_file.path, local_file)
            progress.iter_done_report()

    _download_dir(g.remote_info_dir, g.local_info_dir)

    sly.logger.info("Model has been successfully downloaded")


@sly.timeit
def construct_model_meta():
    class_info_path = os.path.join(g.local_info_dir, 'class_info.json')

    with open(class_info_path, 'r') as class_info_file:
        class_data = json.load(class_info_file)

    h = class_data['color'].lstrip('#')
    rgb = list(int(h[i:i + 2], 16) for i in (0, 2, 4))

    g.meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection([sly.ObjClass(class_data['name'], sly.Rectangle,
                                                                              color=rgb)]))

    return 0


@sly.timeit
def get_model_data():
    return torch.load(g.local_weights_path, map_location='cpu')


@sly.timeit
def init_model():
    model_data = get_model_data()

    model = create_model(model_data['arch'], model_data['heads'], model_data['head_conv'])
    model = load_model(model, g.local_weights_path)

    model = model.to(g.device)
    model.eval()

    return model


@sly.timeit
def deploy_model():
    g.model = init_model()
    sly.logger.info("Model has been successfully deployed")


def download_video(video_id):
    sly.fs.clean_dir(g.input_raw)
    sly.fs.clean_dir(g.input_converted)

    video_info = g.api.video.get_info_by_id(video_id)
    save_path = os.path.join(g.input_raw, video_info.name)

    g.api.video.download_path(video_id, save_path)

    mot_utils.videos_to_frames(save_path)
    return save_path


def process_video(video_id, frames_range):
    download_video(video_id)

    sly_dir_path = os.getcwd()
    os.chdir('../../../src')

    mot_utils.init_script_arguments()

    opt = opts().init()
    inference_model(opt)

    os.chdir(sly_dir_path)


def inference_model(opt):
    model_data = get_model_data()
    model_epoch, model_arch, model_name = model_data['arch'], model_data['heads'], model_data['head_conv']

    opt.load_model = os.path.join(g.local_weights_path)
    opt.arch = model_arch

    output_root = os.path.join(g.output_mot)

    data_type = 'mot'

    video_path = g.video_data['path']
    frame_rate = g.video_data['fps']
    video_index = g.video_data['index']

    dataloader = datasets.LoadImages(video_path, opt.img_size)

    result_filename = os.path.join(output_root, 'tracks', f'{video_index}.txt')
    os.makedirs(os.path.join(output_root, 'tracks'), exist_ok=True)

    eval_seq(opt, dataloader, data_type, result_filename,
             save_dir=output_root, show_image=False, frame_rate=frame_rate,
             epoch=model_epoch)
