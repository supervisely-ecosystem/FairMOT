import torch
import cv2

import os

import supervisely_lib as sly
# from mmcls.apis import init_model
# from mmcv.parallel import collate, scatter
# from mmcls.datasets.pipelines import Compose
from lib.models.model import create_model, load_model
from lib.opts import opts
from sly_eval_seq import eval_seq
import lib.datasets.dataset.jde as datasets

import serve_ann_keeper
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
    sly.logger.info("🟩 Model has been successfully deployed")


def download_frames_interval(video_id, frames_indexes):
    output_path = os.path.join(g.input_converted, f'{video_id}')

    os.makedirs(output_path, exist_ok=True)
    sly.fs.clean_dir(output_path)

    for index, frame_index in enumerate(frames_indexes):
        img_bgr = get_frame_np(g.api, video_id, frame_index)
        cv2.imwrite(f"{output_path}/frame{index:06d}.jpg", img_bgr)  # save frame as JPEG file

    video_data = {'id': video_id, 'path': output_path,
                  'fps': None, 'origin_path': None}

    return video_data


def download_video(video_id, frames_indexes=None):
    save_path = os.path.join(g.input_raw, f'{video_id}.mp4')

    if frames_indexes is None:
        if not os.path.isfile(save_path):
            g.api.video.download_path(video_id, save_path)
        video_data = mot_utils.videos_to_frames(save_path)
    else:
        video_data = download_frames_interval(video_id, frames_indexes)

    return video_data


def get_model_class_name(ann_path=None):
    if ann_path is not None:
        return ann_path.split('/')[-1].split('.')[0].split('_')[-1]

    else:
        class_info_path = os.path.join(g.local_info_dir, 'class_info.json')

        with open(class_info_path, 'r') as class_info_file:
            class_data = json.load(class_info_file)

        return class_data['name']




def process_video(video_id, frames_indexes, conf_thres, is_preview=False,
                  trained_tracker_container=None):
    sly.fs.clean_dir(g.input_converted)  # cleaning dirs before processing
    sly.fs.clean_dir(g.output_mot)

    video_data = download_video(video_id, frames_indexes)

    if trained_tracker_container is not None:
        video_data['fps'] = trained_tracker_container.video_fps

    ann_path, preview_video_path = inference_model(is_preview=is_preview,
                                                   conf_thres=conf_thres,
                                                   frames_indexes=frames_indexes,
                                                   video_data=video_data,
                                                   trained_tracker_container=trained_tracker_container)

    annotations = convert_annotations_to_mot(video_data['path'], ann_path, video_id)
    return annotations, preview_video_path


def upload_video_to_sly(local_video_path):
    remote_video_path = os.path.join("/FairMOT/serve", "preview.mp4")
    if g.api.file.exists(g.team_id, remote_video_path):
        g.api.file.remove(g.team_id, remote_video_path)

    file_info = g.api.file.upload(g.team_id, local_video_path, remote_video_path)

    return file_info

def get_objects_count(ann_path):
    objects_ids = []
    with open(ann_path, 'r') as ann_file:
        ann_rows = ann_file.read().split()

        for ann_row in ann_rows:
            objects_ids.append(ann_row.split(',')[1])

    return len(list(set(objects_ids)))


def get_objects_ids_to_indexes_mapping(ann_path):
    mapping = {}
    indexer = 0

    with open(ann_path, 'r') as ann_file:
        ann_rows = ann_file.read().split()

        for ann_row in ann_rows:
            curr_id = ann_row.split(',')[1]

            rc = mapping.get(curr_id, -1)
            if rc == -1:
                mapping[curr_id] = indexer
                indexer += 1

    return mapping


def get_video_shape(video_path):
    zero_image_name = os.listdir(video_path)[0]
    image_shape = cv2.imread(os.path.join(video_path, zero_image_name)).shape

    return tuple([int(image_shape[1]), int(image_shape[0])])


def get_coords_by_row(row_data, video_shape):
    left, top, w, h = float(row_data[2]), float(row_data[3]), \
                      float(row_data[4]), float(row_data[5])

    bottom = top + h
    if round(bottom) >= video_shape[1] - 1:
        bottom = video_shape[1] - 2
    right = left + w
    if round(right) >= video_shape[0] - 1:
        right = video_shape[0] - 2
    if left < 0:
        left = 0
    if top < 0:
        top = 0

    if right <= 0 or bottom <= 0 or left >= video_shape[0] or top >= video_shape[1]:
        return None
    else:
        return sly.Rectangle(top, left, bottom, right)


def add_figures_from_mot_to_sly(ann_path, ann_keeper, video_shape):
    ids_to_indexes_mapping = get_objects_ids_to_indexes_mapping(ann_path)

    with open(ann_path, 'r') as ann_file:
        ann_rows = ann_file.read().split()

    keep_reading = True if len(ann_rows) > 0 else False
    current_row = 0

    while keep_reading:
        coords_data = []
        objects_indexes = []

        curr_frame_index = ann_rows[current_row].split(',')[0]
        while curr_frame_index == ann_rows[current_row].split(',')[0]:
            row_data = ann_rows[current_row].split(',')

            object_coords = get_coords_by_row(row_data, video_shape=video_shape)
            if object_coords:
                coords_data.append(object_coords)
                objects_indexes.append(ids_to_indexes_mapping[row_data[1]])

            current_row += 1
            if current_row == len(ann_rows):
                keep_reading = False
                break

        ann_keeper.add_figures_by_frame(coords_data=coords_data,
                                        objects_indexes=objects_indexes,
                                        frame_index=curr_frame_index)


def convert_annotations_to_mot(images_seq_path, ann_path, video_id):
    class_name = get_model_class_name(ann_path)
    objects_count = get_objects_count(ann_path)
    video_shape = get_video_shape(images_seq_path)
    video_frames_count = g.api.video.get_info_by_id(video_id).frames_count

    ann_keeper = serve_ann_keeper.AnnotationKeeper(video_shape=(video_shape[1], video_shape[0]),
                                                   objects_count=objects_count,
                                                   class_name=class_name,
                                                   video_frames_count=video_frames_count)

    add_figures_from_mot_to_sly(ann_path=ann_path,
                                ann_keeper=ann_keeper,
                                video_shape=video_shape)

    return ann_keeper.get_annotation()


def generate_video_from_frames():
    output_video_path = os.path.join(g.output_mot, f'preview.mp4')

    cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v libx264 {}'.format(g.output_mot, output_video_path)
    os.system(cmd_str)

    for file in os.listdir(g.output_mot):
        if file.endswith('.jpg'):
            os.remove(os.path.join(g.output_mot, file))

    return output_video_path


def init_options(conf_thres):
    opt = opts().init()
    model_data = get_model_data()
    model_epoch, model_arch, model_heads, model_head_conv = model_data['epoch'], \
                                                            model_data['arch'], \
                                                            model_data['heads'], \
                                                            model_data['head_conv']

    opt.arch = model_arch
    opt.heads = model_heads
    opt.head_conv = model_head_conv
    opt.conf_thres = conf_thres

    opt.load_model = os.path.join(g.local_weights_path)

    return opt


def inference_model(is_preview=False, conf_thres=0, video_data=None,
                    trained_tracker_container=None, frames_indexes=None,):
    mot_utils.init_script_arguments()
    opt = init_options(conf_thres)
    data_type = 'mot'

    frames_save_dir = None
    preview_video_path = None

    class_name = trained_tracker_container.class_name if \
        trained_tracker_container is not None else get_model_class_name()

    video_path, frame_rate, video_id = video_data['path'], video_data['fps'], video_data['id']
    annotations_path = os.path.join(g.output_mot, f'{video_id}_{class_name}.txt')
    os.makedirs(g.output_mot, exist_ok=True)
    dataloader = datasets.LoadImages(video_path, opt.img_size)

    if is_preview:
        frames_save_dir = g.output_mot

    eval_seq(opt, dataloader, data_type, annotations_path,
             save_dir=frames_save_dir, frame_rate=frame_rate,
             frames_indexes=frames_indexes,
             trained_tracker_container=trained_tracker_container)

    if is_preview:
        preview_video_path = generate_video_from_frames()

    return annotations_path, preview_video_path


def get_frame_np(api, video_id, frame_index):
    img_rgb = api.video.frame.download_np(video_id, frame_index)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def validate_figure(img_height, img_width, figure):
    img_size = (img_height, img_width)
    # check figure is within image bounds
    canvas_rect = sly.Rectangle.from_size(img_size)
    if canvas_rect.contains(figure.to_bbox()) is False:
        # crop figure
        figures_after_crop = [cropped_figure for cropped_figure in figure.crop(canvas_rect)]
        if len(figures_after_crop) != 1:
            g.logger.warn("len(figures_after_crop) != 1")
        return figures_after_crop[0]
    else:
        return figure


def calculate_nofity_step(frames_forward):
    if frames_forward > 40:
        return 10
    else:
        return 5
