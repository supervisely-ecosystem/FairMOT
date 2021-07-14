import os
from collections import namedtuple
import shelve
import supervisely_lib as sly
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress

from supervisely_lib.io.fs import mkdir, get_file_name

from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.geometry.rectangle import Rectangle

import shutil

progress_index = 1
_images_infos = None # dataset_name -> image_name -> image_info
_cache_base_filename = os.path.join(g.my_app.data_dir, "images_info")
_cache_path = _cache_base_filename + ".db"
_image_id_to_paths = {}


def init(data, state):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)

    init_progress(progress_index, data)
    init_progress("InputVideo", data)
    init_progress("InputFrames", data)

    data["done1"] = False
    state["collapsed1"] = False

    state["validationTeamId"] = None
    state["validationWorkspaceId"] = None
    state["validationProjectId"] = None
    state["validationDatasets"] = []
    state["validationAllDatasets"] = True



"""
RUN mkdir /FairMOT/data && cd /FairMOT/data &&  wget $dataset_link && tar -xvf *.tar && mv ds_* SLY_MOT
RUN rm -f -r /FairMOT/data/*.tar && cd /FairMOT/data/SLY_MOT/ && mkdir -p labels_with_ids/train && mkdir images && mkdir -p labels_with_ids/train && cp -r train images/test && mv train images

RUN cd /FairMOT/ && git pull

RUN pip install -r /FairMOT/requirements.txt 
# RUN python /FairMOT/src/rename_img_seq.py
RUN python /FairMOT/src/gen_labels.py
RUN cd /FairMOT/src/ && python ./gen_data_path.py

"""


@g.my_app.callback("download_project_train")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def download_project_train(api: sly.api, task_id, context, state, app_logger):
    download_project([g.project_id])


@g.my_app.callback("skip_validation")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def download_project_validation(api: sly.api, task_id, context, state, app_logger):
    # download_project()
    fields = [
        {"field": f"data.done2", "payload": True},
        {"field": f"state.collapsed3", "payload": False},
        {"field": f"state.disabled3", "payload": False},
        {"field": f"state.activeStep", "payload": 3},
    ]
    g.api.app.set_fields(g.task_id, fields)


def download_project(project_ids):
    try:
        if not sly.fs.dir_exists(g.project_dir):
            sly.fs.mkdir(g.project_dir)

        from_sl_to_MOT(projects_ids=project_ids)
        # organize_in_mot_format(is_train)

    except Exception as e:
        raise e

    fields = [
        {"field": f"data.done1", "payload": True},
        {"field": f"state.collapsed2", "payload": False},
        {"field": f"state.disabled2", "payload": False},
        {"field": f"state.activeStep", "payload": 2},
    ]
    g.api.app.set_field(g.task_id, "data.scrollIntoView", f"step{2}")
    g.api.app.set_fields(g.task_id, fields)


def from_sl_to_MOT(projects_ids):
    images_dir_name = 'img1'
    ann_dir_name = 'gt'
    # dir_train = 'train'
    image_ext = '.jpg'
    seq_name = 'seqinfo.ini'
    frame_rate = 25  # @TODO: from video
    # video_info.frames_to_timecodes.index(1)

    conf_tag_name = 'ignore_conf'

    download_progress_project = get_progress_cb("InputProject", "Current project", len(projects_ids))

    for project_id in projects_ids:
        project = g.api.project.get_info_by_id(project_id)
        if project is None:
            raise RuntimeError("Project with ID {!r} not found".format(g.project_id))
        if project.type != str(sly.ProjectType.VIDEOS):
            raise TypeError("Project type is {!r}, but have to be {!r}".format(project.type, sly.ProjectType.VIDEOS))

        meta_json = g.api.project.get_meta(project_id)
        meta = sly.ProjectMeta.from_json(meta_json)

        result_dir = os.path.join(g.my_app.data_dir, f'input_data',
                                  str(project_id))

        key_id_map = KeyIdMap()
        project_datasets = g.api.dataset.get_list(project_id)
        download_progress_dataset = get_progress_cb("InputDataset", "Current dataset", len(project_datasets))
        for dataset in project_datasets:
            videos = g.api.video.get_list(dataset.id)
            download_progress_videos = get_progress_cb("InputVideo", "Current video", len(videos))
            for batch in sly.batched(videos, batch_size=10):
                for video_info in batch:

                    ann_info = g.api.video.annotation.download(video_info.id)
                    ann = sly.VideoAnnotation.from_json(ann_info, meta, key_id_map)
                    curr_objs_geometry_types = [obj.obj_class.geometry_type for obj in ann.objects]

                    if sly.Rectangle not in curr_objs_geometry_types:
                        print('invalid_shape')
                        continue

                    result_images = os.path.join(result_dir, dataset.name, get_file_name(video_info.name),
                                                 images_dir_name)
                    result_anns = os.path.join(result_dir, dataset.name, get_file_name(video_info.name),
                                               ann_dir_name)
                    seq_path = os.path.join(result_dir, dataset.name, get_file_name(video_info.name), seq_name)

                    # gt_path = os.path.join(result_anns, gt_name)

                    mkdir(result_images)
                    mkdir(result_anns)

                    with open(seq_path, 'a') as f:
                        f.write('[Sequence]\n')
                        f.write('name={}\n'.format(get_file_name(video_info.name)))
                        f.write('imDir={}\n'.format(images_dir_name))
                        f.write('frameRate={}\n'.format(round(1 / video_info.frames_to_timecodes[1])))
                        f.write('seqLength={}\n'.format(video_info.frames_count))
                        f.write('imWidth={}\n'.format(video_info.frame_width))
                        f.write('imHeight={}\n'.format(video_info.frame_height))
                        f.write('imExt={}\n'.format(image_ext))

                    id_to_video_obj = {}
                    for idx, curr_video_obj in enumerate(ann.objects):
                        id_to_video_obj[curr_video_obj] = idx + 1


                    download_progress_frames = get_progress_cb("InputFrames", "Downloading frames",
                                                               len(ann.frames))

                    for frame_index, frame in enumerate(ann.frames):
                        for figure in frame.figures:

                            if figure.video_object.obj_class.geometry_type != Rectangle:
                                continue

                            rectangle_geom = figure.geometry.to_bbox()
                            left = rectangle_geom.left
                            top = rectangle_geom.top
                            width = rectangle_geom.width
                            height = rectangle_geom.height
                            conf_val = 1
                            for curr_tag in figure.video_object.tags:
                                if conf_tag_name == curr_tag.name and (
                                        curr_tag.frame_range is None or frame_index in range(curr_tag.frame_range[0],
                                                                                             curr_tag.frame_range[1] + 1)):
                                    conf_val = 0
                            curr_gt_data = '{},{},{},{},{},{},{},{},{},{}\n'.format(frame_index + 1,
                                                                                    id_to_video_obj[figure.video_object],
                                                                                    left, top, width - 1, height - 1,
                                                                                    conf_val, -1, -1, -1)
                            filename = 'gt_{}.txt'.format(figure.parent_object.obj_class.name)
                            with open(os.path.join(result_anns, filename), 'a') as f:  # gt_path
                                f.write(curr_gt_data)
                        image_name = str(frame_index + 1).zfill(6) + image_ext
                        image_path = os.path.join(result_images, image_name)
                        if frame_index == ann.frames_count:
                            break
                        g.api.video.frame.download_path(video_info.id, frame_index, image_path)
                        download_progress_frames(1)  # updating progress
                    download_progress_videos(1)
            download_progress_dataset(1)
        download_progress_project(1)

    reset_progress("InputProject")
    reset_progress("InputDataset")
    reset_progress("InputVideo")
    reset_progress("InputFrames")

