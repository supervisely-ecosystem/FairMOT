import os
import supervisely_lib as sly
import sly_globals as g
import random


def restart(data, state):
    data["done3"] = False


def refresh_table(total_items_count):

    train_percent = 80
    train_count = int(total_items_count / 100 * train_percent)
    random_split_tab = {
        "count": {
            "total": total_items_count,
            "train": train_count,
            "val": total_items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareVideosBetweenSplits": False,
        "sliderDisabled": False,
    }

    g.api.app.set_fields(g.task_id,
        [
            {'field': 'state.randomSplit', 'payload': random_split_tab},
            {'field': 'data.totalVideosCount', 'payload': total_items_count},
        ]
    )


def init(data, state):

    state["splitMethod"] = "random"

    data["randomSplit"] = [
        {"name": "train", "type": "success"},
        {"name": "val", "type": "primary"},
        {"name": "total", "type": "gray"},
    ]

    state["randomSplit"] = []
    data['data.totalVideosCount'] = 0

    # refresh_table(g.project_info.items_count)
    # state["trainTagName"] = ""
    # if project_meta.tag_metas.get("train") is not None:
    #     state["trainTagName"] = "train"
    # state["valTagName"] = ""
    # if project_meta.tag_metas.get("val") is not None:
    #     state["valTagName"] = "val"

    state["trainDatasets"] = []
    state["valDatasets"] = []
    state["untaggedVideos"] = "train"
    state["splitInProgress"] = False
    data["trainVideosCount"] = None
    data["valVideosCount"] = None
    data["done3"] = False
    state["collapsed3"] = not True
    state["disabled3"] = not True

    state["trainVideosPaths"] = None
    state["valVideosPaths"] = None


def get_train_val_sets(state):
    split_method = state["splitMethod"]
    if split_method == "random":
        train_count = state["randomSplit"]["count"]["train"]
        val_count = state["randomSplit"]["count"]["val"]

        train_videos_paths, val_videos_paths = split_videos_randomly_by_counts(train_count, val_count)
        return train_videos_paths, val_videos_paths

    elif split_method == "datasets":
        train_datasets_names = state["trainDatasets"]
        val_datasets_names = state["valDatasets"]
        train_videos_paths, val_videos_paths = split_videos_by_datasets(train_datasets_names, val_datasets_names)
        return train_videos_paths, val_videos_paths

    # elif split_method == "tags":
    #     train_tag_name = state["trainTagName"]
    #     val_tag_name = state["valTagName"]
    #     add_untagged_to = state["untaggedVideos"]
    #     train_set, val_set = sly.Project.get_train_val_splits_by_tag(project_dir, train_tag_name, val_tag_name,
    #                                                                  add_untagged_to)
    #     return train_set, val_set
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")


def split_videos_randomly_by_counts(train_count, val_count):
    train_videos_paths = []
    val_videos_paths = []

    counts = {'train': train_count,
              'val': val_count}

    ds_paths = get_ds_paths()
    for ds_path in ds_paths:
        train_videos_paths_temp, \
        val_videos_paths_temp = get_video_paths_by_ds_and_counts(counts, ds_path)

        train_videos_paths.extend(train_videos_paths_temp)
        val_videos_paths.extend(val_videos_paths_temp)

    return train_videos_paths, val_videos_paths


def get_video_paths_by_ds_and_counts(counts, ds_path):
    train_videos_paths = []
    val_videos_paths = []
    class_label = g.api.app.get_field(g.task_id, 'state.selectedClass')

    video_names = [name for name in os.listdir(ds_path) if  # filter videos by selected class
                   len(g.get_files_paths(os.path.join(ds_path, name), [f'{class_label}.txt'])) > 0]

    for video_name in video_names:
        if random.choice([True, False]):
            if counts['train'] > 0:
                train_videos_paths.append(video_name)
                counts['train'] -= 1
            elif counts['val'] > 0:
                val_videos_paths.append(video_name)
                counts['val'] -= 1
        else:
            if counts['val'] > 0:
                val_videos_paths.append(video_name)
                counts['val'] -= 1
            elif counts['train'] > 0:
                train_videos_paths.append(video_name)
                counts['train'] -= 1

    return [os.path.join(ds_path, curr_video_name) for curr_video_name in train_videos_paths], \
           [os.path.join(ds_path, curr_video_name) for curr_video_name in val_videos_paths]


def split_videos_by_datasets(train_datasets_names, val_datasets_names):
    train_videos_paths = []
    val_videos_paths = []

    ds_paths = get_ds_paths()

    for ds_path in ds_paths:
        if ds_path.split('/')[-1] in train_datasets_names:
            train_videos_paths.extend(get_video_paths_by_ds_and_counts({'train': 9999,
                                                                        'val': 0}, ds_path)[0])
        if ds_path.split('/')[-1] in val_datasets_names:
            val_videos_paths.extend(get_video_paths_by_ds_and_counts({'train': 0,
                                                                      'val': 9999}, ds_path)[1])

    return train_videos_paths, val_videos_paths


def get_ds_paths(projects_ids=None):
    ds_paths = []

    input_data_path = os.path.join(g.my_app.data_dir, 'input_data_mot')
    projects_ids = sorted(
        [name for name in os.listdir(input_data_path) if os.path.isdir(os.path.join(input_data_path, name))])

    for project_id in projects_ids:
        project_path = os.path.join(input_data_path, project_id)
        dataset_names = [name for name in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, name))]

        for ds_name in dataset_names:
            ds_paths.append(os.path.join(project_path, ds_name))

    return ds_paths


@g.my_app.callback("create_splits")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def create_splits(api: sly.Api, task_id, context, state, app_logger):
    step_done = False
    train_videos_paths = None
    val_videos_paths = None
    try:
        api.task.set_field(task_id, "state.splitInProgress", True)
        train_videos_paths, val_videos_paths = get_train_val_sets(state)
        sly.logger.info(f"Train set: {len(train_videos_paths)} videos")
        sly.logger.info(f"Val set: {len(val_videos_paths)} videos")
        verify_train_val_sets(train_videos_paths, val_videos_paths)
        step_done = True
    except Exception as e:
        train_videos_paths = None
        val_videos_paths = None
        step_done = False
        raise e
    finally:
        api.task.set_field(task_id, "state.splitInProgress", False)
        fields = [
            {"field": "state.splitInProgress", "payload": False},
            {"field": f"data.done3", "payload": step_done},
            {"field": f"data.trainVideosCount",
             "payload": None if train_videos_paths is None else len(train_videos_paths)},
            {"field": f"data.valVideosCount", "payload": None if val_videos_paths is None else len(val_videos_paths)},
        ]
        if step_done is True:
            fields.extend([
                {"field": "state.collapsed4", "payload": False},
                {"field": "state.disabled4", "payload": False},
                {"field": "state.activeStep", "payload": 4},
                {"field": "state.trainVideosPaths", "payload": train_videos_paths, "append": True, "recursive": False},
                {"field": "state.valVideosPaths", "payload": val_videos_paths, "append": True, "recursive": False},

            ])
            g.api.app.set_field(g.task_id, "data.scrollIntoView", f"step{4}")
        g.api.app.set_fields(g.task_id, fields)
