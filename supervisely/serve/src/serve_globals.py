import os
import sys
import pathlib
import supervisely_lib as sly


my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

train_source_path = os.path.join(root_source_path, "supervisely/train/src")
sly.logger.info(f"Train source directory: {train_source_path}")
sys.path.append(train_source_path)

serve_source_path = os.path.join(root_source_path, "supervisely/serve/src")
sly.logger.info(f"Serve source directory: {serve_source_path}")
sys.path.append(serve_source_path)

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
remote_weights_path = os.environ['modal.state.slyFile']
device = os.environ['modal.state.device']

remote_exp_dir = str(pathlib.Path(remote_weights_path).parents[1])
remote_info_dir = os.path.join(remote_exp_dir, "info")

local_info_dir = os.path.join(my_app.data_dir, "info")
sly.fs.mkdir(local_info_dir)

input_raw = os.path.join(my_app.data_dir, "input_raw")
sly.fs.mkdir(input_raw)

input_converted = os.path.join(my_app.data_dir, "input_converted")
sly.fs.mkdir(input_converted)

output_mot = os.path.join(my_app.data_dir, "output_mot")
sly.fs.mkdir(output_mot)

local_weights_path = os.path.join(my_app.data_dir, sly.fs.get_file_name_with_ext(remote_weights_path))

model = None
video_data = None
meta: sly.ProjectMeta = None


def get_files_paths(src_dir, extensions):
    files_paths = []
    for root, dirs, files in os.walk(src_dir):
        for extension in extensions:
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    files_paths.append(file_path)

    return files_paths
