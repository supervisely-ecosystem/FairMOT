import os
from pathlib import Path
import sys
import supervisely_lib as sly
import pickle

from dotenv import load_dotenv  # pip install python-dotenv\

load_dotenv("../debug.env")
load_dotenv("../secret_debug.env", override=True)

my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

project_info = api.project.get_info_by_id(project_id)
if project_info is None:  # for debug
    raise ValueError(f"Project with id={project_id} not found")

# sly.fs.clean_dir(my_app.data_dir)  # for debug

project_dir = os.path.join(my_app.data_dir, "visualize_fairMOT")
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

model_meta = None

temp_files = os.path.join(project_dir, "temp_files")

if os.path.exists(temp_files):  # clean temp
    sly.fs.clean_dir(temp_files)

converted_dir = os.path.join(temp_files, "converted_input")
sly.fs.mkdir(converted_dir)

projects_dir = os.path.join(temp_files, "projects")
sly.fs.mkdir(projects_dir)
checkpoints_dir = os.path.join(temp_files, "checkpoints")
sly.fs.mkdir(checkpoints_dir)
local_info_dir = os.path.join(temp_files, "info")
sly.fs.mkdir(local_info_dir)

grid_video_dir = os.path.join(temp_files, "grid_video")
sly.fs.mkdir(grid_video_dir)

output_dir = os.path.join(project_dir, "output_results")

if os.path.exists(output_dir):  # clean output
    sly.fs.clean_dir(output_dir)
else:
    sly.fs.mkdir(output_dir)



root_source_dir = str(Path(os.path.abspath(sys.argv[0])).parents[3])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)
source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"App source directory: {source_path}")
sys.path.append(source_path)
ui_sources_dir = os.path.join(source_path, "ui")
sly.logger.info(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")
fair_mot_src = os.path.join(root_source_dir, "src")
sys.path.append(fair_mot_src)


def dump_req(req_objects, filename):
    save_path = os.path.join(my_app.data_dir, 'dumps')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, filename)
    with open(save_path, 'wb') as dump_file:
        pickle.dump(req_objects, dump_file)


def load_dumped(filename):
    load_path = os.path.join(my_app.data_dir, 'dumps', filename)
    with open(load_path, 'rb') as dumped:
        return pickle.load(dumped)

def get_files_paths(src_dir, extensions):
    files_paths = []
    for root, dirs, files in os.walk(src_dir):
        for extension in extensions:
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    files_paths.append(file_path)

    return files_paths
