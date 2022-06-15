import supervisely as sly
import sly_globals as g

import os
from functools import partial
from sly_train_progress import get_progress_cb, reset_progress, init_progress
from input_data import object_ann_info

import splits


def init(data, state):
    project_meta = g.api.project.get_meta(id=g.project_id)

    rows_names = generate_rows_by_ann(project_meta)
    state['selectedClass'] = None

    for row_name in rows_names:
        if not row_name['isDisabled']:
            state['selectedClass'] = row_name['name']
            break

    rows_names = sorted(rows_names, key=lambda k: k['isDisabled'])

    data["selectClassTable"] = rows_names
    data["classDatasets"] = []

    state["statsLoaded"] = False
    state["loadingStats"] = False

    data["done2"] = False
    state["collapsed2"] = True
    state["disabled2"] = True


def restart(data, state):
    data['done2'] = False


def generate_rows_by_ann(ann_meta):
    rows = []

    for index, curr_class in enumerate(ann_meta['classes']):
        rows.append({
            "name": f"{curr_class['title']}",
            "shapeType": f"{curr_class['shape']}",
            "color": f"{curr_class['color']}",
            "isDisabled": False if curr_class['shape'] == 'rectangle' else True,
        })

    return rows


def get_datasets_list_by_class_label(class_label):
    datasets_names_list = []

    input_data_dir = os.path.join(g.my_app.data_dir, f'input_data_mot')

    available_ds_paths = g.get_files_paths(input_data_dir, [f'{class_label}.txt'])

    for available_ds_path in available_ds_paths:
        root_path = available_ds_path.split(g.my_app.data_dir)[-1]
        ds_name = root_path.split('/')[3]
        datasets_names_list.append(ds_name)

    return list(set(datasets_names_list))


def generate_selector_list(dataset_names):
    selector_list = []

    for dataset_name in dataset_names:
        selector_list.append(
            {
                "label": dataset_name,
                "value": dataset_name
            }
        )
    return selector_list


@g.my_app.callback("select_class")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_class(api: sly.api, task_id, context, state, app_logger):
    table = g.api.app.get_field(g.task_id, 'data.selectClassTable')
    class_label = state['selectedClass']
    total_videos_count = [row['labeledVideosCount'] for row in table if row['name'] == class_label][0]

    class_datasets_list = get_datasets_list_by_class_label(class_label)
    selector_list = generate_selector_list(class_datasets_list)

    splits.refresh_table(total_videos_count)

    fields = [
        {"field": "data.done2", "payload": True},
        {"field": "state.collapsed3", "payload": False},
        {"field": "state.disabled3", "payload": False},
        {"field": "state.activeStep", "payload": 3},
        {"field": "data.classDatasets", "payload": selector_list},

    ]
    g.api.app.set_field(g.task_id, "data.scrollIntoView", f"step{3}")
    api.task.set_fields(task_id, fields)


@g.my_app.callback("load_objects_stats")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def load_objects_stats(api: sly.api, task_id, context, state, app_logger):
    objects_ann_info = g.load_dumped('ann_info.pkl')
    table = g.api.app.get_field(g.task_id, 'data.selectClassTable')

    fill_tables_by_video_objects(state, objects_ann_info, table)

    fields = [
        {"field": "state.loadingStats", "payload": False},
        {"field": "state.statsLoaded", "payload": True},
    ]
    api.task.set_fields(task_id, fields)


def add_counters_to_dict(curr_dict, labels):
    for label_on_frame in labels:
        curr_dict[label_on_frame] += 1


def fill_tables_by_video_objects(state, objects_ann_info, table):
    class_id_to_labels_mapping = {}

    for curr_project in objects_ann_info.values():  # extract objects ids to labels
        for curr_dataset in curr_project.values():
            for curr_video in curr_dataset:
                objects_on_video = curr_video['objects']
                for curr_object in objects_on_video:
                    class_id_to_labels_mapping[curr_object['id']] = curr_object['classTitle']

    labeled_videos_count = {label: 0 for label in class_id_to_labels_mapping.values()}
    labeled_frames_count = {label: 0 for label in class_id_to_labels_mapping.values()}
    labeled_objects_count = {label: 0 for label in class_id_to_labels_mapping.values()}

    for curr_project in objects_ann_info.values():
        for curr_dataset in curr_project.values():
            for curr_video in curr_dataset:
                labels_on_video = []

                for curr_frame in curr_video['frames']:
                    figures_on_frame = curr_frame['figures']
                    labels_on_frame = [class_id_to_labels_mapping[figure_on_frame['objectId']]
                                       for figure_on_frame in figures_on_frame]  # for count objects
                    add_counters_to_dict(labeled_objects_count, labels_on_frame)

                    labels_on_frame = list(set(labels_on_frame))  # for count frames
                    add_counters_to_dict(labeled_frames_count, labels_on_frame)

                    labels_on_video.extend(labels_on_frame)  # for count video

                labels_on_video = list(set(labels_on_video))
                add_counters_to_dict(labeled_videos_count, labels_on_video)

    for row in table:
        row['labeledVideosCount'] = labeled_videos_count.get(row['name'], 0)
        row['labeledFramesCount'] = labeled_frames_count.get(row['name'], 0)
        row['labeledObjectsCount'] = labeled_objects_count.get(row['name'], 0)

    fields = [
        {"field": f"data.selectClassTable", "payload": table, "recursive": False},
    ]
    g.api.task.set_fields(g.task_id, fields)

    return 0
