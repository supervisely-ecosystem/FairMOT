import supervisely_lib as sly
import sly_globals as g

import os
from functools import partial

from sly_visualize_progress import get_progress_cb, reset_progress, init_progress
from input_data import object_ann_info


def init(data, state):
    state['selectedClass'] = None

    data["modelsTable"] = []


    state["statsLoaded"] = False
    state["loadingStats"] = False
    state["selectedModels"] = ['model_last.pth']

    data["done3"] = False
    state["collapsed3"] = True
    state["disabled3"] = True


def restart(data, state):
    data['done3'] = False


@g.my_app.callback("apply_models")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_models(api: sly.api, task_id, context, state, app_logger):
    selected_count = len(state['selectedModels'])

    if selected_count == 0:
        raise ValueError('No models selected. Please select models.')

    fields = [
        {"field": "data.done3", "payload": True},
        {"field": "state.collapsed4", "payload": False},
        {"field": "state.disabled4", "payload": False},
        {"field": "state.activeStep", "payload": 4},

    ]

    api.task.set_fields(task_id, fields)
