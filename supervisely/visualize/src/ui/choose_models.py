import supervisely_lib as sly
import sly_globals as g

import os
from functools import partial

from sly_train_progress import get_progress_cb, reset_progress, init_progress
from input_data import object_ann_info


def init(data, state):
    state['selectedClass'] = None

    data["modelsTable"] = []

    state["statsLoaded"] = False
    state["loadingStats"] = False

    data["done3"] = False
    state["collapsed3"] = not True
    state["disabled3"] = not True


def restart(data, state):
    data['done3'] = False


@g.my_app.callback("apply_models")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_models(api: sly.api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.done3", "payload": True},
        {"field": "state.collapsed4", "payload": False},
        {"field": "state.disabled4", "payload": False},
        {"field": "state.activeStep", "payload": 4},

    ]

    api.task.set_fields(task_id, fields)


@g.my_app.callback("select_all")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_models(api: sly.api, task_id, context, state, app_logger):
    change_selection(flag=True)


@g.my_app.callback("deselect_all")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_models(api: sly.api, task_id, context, state, app_logger):
    change_selection(flag=False)


def change_selection(flag=True):
    table_rows = g.api.app.get_field(g.task_id, 'data.modelsTable')
    for row in table_rows:
        row['selected'] = flag
    fields = [
        {"field": f"data.modelsTable", "payload": table_rows, "recursive": False},
    ]
    g.api.task.set_fields(g.task_id, fields)
