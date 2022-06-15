import supervisely as sly
import sly_globals as g

import os

from sly_train_progress import get_progress_cb, reset_progress, init_progress

from supervisely_lib.io.fs import mkdir, get_file_name

from supervisely_lib.geometry.rectangle import Rectangle

import cv2
from glob import glob


progress_index = 1
object_ann_info = None


def init(data, state):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectItemsCount"] = g.project_info.items_count if g.project_info.items_count else 0
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)

    init_progress("InputProject", data)
    init_progress("InputDataset", data)
    init_progress("InputVideo", data)
    init_progress("InputFrames", data)

    data["done1"] = False
    state["collapsed1"] = False

    state["validationTeamId"] = None
    state["validationWorkspaceId"] = None
    state["validationProjectId"] = None
    state["validationDatasets"] = []
    state["validationAllDatasets"] = True


def download_projects_sly(projects_ids):
    try:
        if not sly.fs.dir_exists(g.project_dir):
            sly.fs.mkdir(g.project_dir)

        download_progress_project = get_progress_cb("InputProject", "Downloading project", len(projects_ids))
        for project_id in projects_ids:
            dest_dir = os.path.join(g.my_app.data_dir, f'input_data_sly', str(project_id))
            sly.download_video_project(g.api, project_id, dest_dir, log_progress=True)
            download_progress_project(1)

        download_progress_project = get_progress_cb("InputProject", "Downloading annotations", len(projects_ids))
        projects_ann_info = {}
        for project_id in projects_ids:
            projects_ann_info[project_id] = get_project_ann_info(project_id, all_ds=True)
            download_progress_project(1)
        g.dump_req(projects_ann_info, 'ann_info.pkl')

        reset_progress("InputProject")
    except Exception as e:
        raise e


def convert_projects_mot(projects_ids):
    progress_project = get_progress_cb("InputProject", "Converting to MOT", len(projects_ids))
    for project_id in projects_ids:
        input_project_path = os.path.join(g.my_app.data_dir, f'input_data_sly', str(project_id))
        converted_project_path = os.path.join(g.my_app.data_dir, f'input_data_mot', str(project_id))
        convert_project(input_project_path, converted_project_path)
        progress_project(1)
    reset_progress('InputProject')


@g.my_app.callback("download_project_train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_project_train(api: sly.api, task_id, context, state, app_logger):
    download_projects_sly([g.project_id])
    convert_projects_mot([g.project_id])

    fields = [
        {"field": f"data.done1", "payload": True},
        {"field": f"state.collapsed2", "payload": False},
        {"field": f"state.disabled2", "payload": False},
        {"field": f"state.activeStep", "payload": 2},
    ]
    g.api.app.set_field(g.task_id, "data.scrollIntoView", f"step{2}")
    g.api.app.set_fields(g.task_id, fields)


def convert_project(input_project_path, output_project_path):
    images_dir_name = 'img1'
    ann_dir_name = 'gt'
    # dir_train = 'train'
    image_ext = '.jpg'
    seq_name = 'seqinfo.ini'

    datasets_pathes = glob(input_project_path + "/*/")
    if len(datasets_pathes) == 0:
        raise FileExistsError('There is no datasets in project')

    meta_json = sly.io.json.load_json_file(os.path.join(input_project_path, 'meta.json'))
    meta = sly.ProjectMeta.from_json(meta_json)
    progress_dataset = get_progress_cb("InputDataset", "Current dataset", len(datasets_pathes))
    for ds_path in datasets_pathes:
        ds_name = ds_path.split('/')[-2]
        anns_pathes = glob(ds_path + "ann" + "/*")

        progress_videos = get_progress_cb("InputVideo", "Current video", len(anns_pathes))
        for ann_path in anns_pathes:
            ann_json = sly.io.json.load_json_file(ann_path)
            ann = sly.VideoAnnotation.from_json(ann_json, meta)
            video_name = sly.io.fs.get_file_name(ann_path)
            video_path = os.path.join(ds_path, "video", video_name)
            video_info = sly.video.get_info(video_path)['streams'][0]

            curr_objs_geometry_types = [obj.obj_class.geometry_type for obj in ann.objects]

            result_images = os.path.join(output_project_path, ds_name, video_name,
                                         images_dir_name)
            result_anns = os.path.join(output_project_path, ds_name, video_name,
                                       ann_dir_name)
            seq_path = os.path.join(output_project_path, ds_name, video_name, seq_name)


            mkdir(result_images)
            mkdir(result_anns)

            with open(seq_path, 'a') as f:
                f.write('[Sequence]\n')
                f.write('name={}\n'.format(get_file_name(video_name)))
                f.write('imDir={}\n'.format(images_dir_name))
                f.write('frameRate={}\n'.format(round(1 / video_info['framesToTimecodes'][1])))
                f.write('seqLength={}\n'.format(video_info['framesCount']))
                f.write('imWidth={}\n'.format(video_info['width']))
                f.write('imHeight={}\n'.format(video_info['height']))
                f.write('imExt={}\n'.format(image_ext))

            id_to_video_obj = {}
            for idx, curr_video_obj in enumerate(ann.objects):
                id_to_video_obj[curr_video_obj] = idx + 1

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
                    curr_gt_data = '{},{},{},{},{},{},{},{},{},{}\n'.format(frame.index + 1,
                                                                            id_to_video_obj[figure.video_object],
                                                                            left, top, width - 1, height - 1,
                                                                            conf_val, -1, -1, -1)
                    filename = 'gt_{}.txt'.format(figure.parent_object.obj_class.name)
                    with open(os.path.join(result_anns, filename), 'a') as f:  # gt_path
                        f.write(curr_gt_data)
                if frame_index == ann.frames_count:
                    break

            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            count = 1
            while success:
                image_name = str(count).zfill(6) + image_ext
                image_path = os.path.join(result_images, image_name)
                cv2.imwrite(image_path, image)
                success, image = vidcap.read()
                count += 1

            progress_videos(1)
        progress_dataset(1)

    reset_progress('InputVideo')
    reset_progress('InputDataset')


def get_project_ann_info(project_id, dataset_names=None, all_ds=True):
    ann_infos = {}

    if all_ds:
        dataset_ids = [ds.id for ds in g.api.dataset.get_list(project_id)]
    else:
        dataset_ids = [ds.id for ds in g.api.dataset.get_list(project_id) if ds.name in dataset_names]

    sly_progress_ds = get_progress_cb("InputDataset", "Current dataset", len(dataset_ids))

    for dataset_id in dataset_ids:
        video_list = g.api.video.get_list(dataset_id)
        video_ids = [video_info.id for video_info in video_list]
        sly_progress_ann = get_progress_cb("InputVideo", "Download annotations", 1)

        ann_infos[dataset_id] = g.api.video.annotation.download_bulk(dataset_id, video_ids)

        sly_progress_ann(1)
        sly_progress_ds(1)

    reset_progress('progressInputDataset')
    reset_progress('progressInputVideo')
    return ann_infos
