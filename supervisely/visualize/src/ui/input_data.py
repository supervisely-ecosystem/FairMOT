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


@g.my_app.callback("download_projects_handler")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def download_projects_handler(api: sly.api, task_id, context, state, app_logger):
    download_projects([g.project_id])


def download_projects(project_ids):
    download_progress = get_progress_cb('InputProject', "Download project", len(project_ids))
    for project_id in project_ids:
        try:
            project_dir = os.path.join(g.projects_dir, f'{project_id}')
            if not sly.fs.dir_exists(project_dir):
                sly.fs.mkdir(project_dir)
            sly.download_video_project(g.api, project_id, project_dir, log_progress=True)
        except Exception as e:
            raise e
        download_progress(1)
    reset_progress('InputProject')

    fields = [
        {"field": f"data.done1", "payload": True},
        {"field": f"state.collapsed2", "payload": False},
        {"field": f"state.disabled2", "payload": False},
        {"field": f"state.activeStep", "payload": 2},
    ]
    g.api.app.set_field(g.task_id, "data.scrollIntoView", f"step{2}")
    g.api.app.set_fields(g.task_id, fields)
