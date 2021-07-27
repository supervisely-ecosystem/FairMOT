import os

import supervisely_lib as sly
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress

import cv2

import ffmpeg



progress_index = 1
_images_infos = None  # dataset_name -> image_name -> image_info
_cache_base_filename = os.path.join(g.my_app.data_dir, "images_info")
_cache_path = _cache_base_filename + ".db"
_image_id_to_paths = {}

object_ann_info = None


def init(data, state):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectItemsCount"] = g.project_info.items_count
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)

    init_progress("InputProject", data)

    data["done1"] = False
    state["collapsed1"] = False

    state["validationTeamId"] = None
    state["validationWorkspaceId"] = None
    state["validationProjectId"] = None
    state["validationDatasets"] = []
    state["validationAllDatasets"] = True

    data['videosData'] = []


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


def videos_to_frames(project_path, videos_data):

    videos_paths = g.get_files_paths(project_path, '.mp4')

    for video_index, video_path in enumerate(videos_paths):

        project_id = video_path.split('/')[-4]
        ds_name = video_path.split('/')[-3]
        video_name = (video_path.split('/')[-1]).split('.mp4')[0]
        output_path = os.path.join(g.converted_dir, f'{project_id}_{ds_name}_{video_name}')

        os.makedirs(output_path, exist_ok=True)

        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        rotateCode = check_rotation(video_path)

        while success:
            if rotateCode is not None:
                image = correct_rotation(image, rotateCode)
            cv2.imwrite(f"{output_path}/frame{count:06d}.jpg", image)  # save frame as JPEG file
            success, image = vidcap.read()

            count += 1

        fps = vidcap.get(cv2.CAP_PROP_FPS)

        videos_data.append(
            {'index': video_index, 'path': output_path,
             'fps': fps, 'origin_path': video_path}
        )


@g.my_app.callback("download_projects_handler")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_projects_handler(api: sly.api, task_id, context, state, app_logger):
    download_projects([g.project_id])


def download_projects(project_ids):
    download_progress = get_progress_cb('InputProject', "Download project", len(project_ids))
    videos_data = []
    for project_id in project_ids:
        try:
            project_dir = os.path.join(g.projects_dir, f'{project_id}')
            if not sly.fs.dir_exists(project_dir):
                sly.fs.mkdir(project_dir)
            else:
                sly.fs.clean_dir(project_dir)
            sly.download_video_project(g.api, project_id, project_dir, log_progress=True)
            videos_to_frames(project_dir, videos_data)

        except Exception as e:
            raise e
        download_progress(1)
    reset_progress('InputProject')

    fields = [

        {"field": f"data.videosData", "payload": videos_data},
        {"field": f"data.done1", "payload": True},
        {"field": f"state.collapsed2", "payload": False},
        {"field": f"state.disabled2", "payload": False},
        {"field": f"state.activeStep", "payload": 2},
    ]
    g.api.app.set_field(g.task_id, "data.scrollIntoView", f"step{2}")
    g.api.app.set_fields(g.task_id, fields)
