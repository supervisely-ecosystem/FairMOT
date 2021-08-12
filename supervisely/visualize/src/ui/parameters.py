import os
import supervisely_lib as sly
import sly_globals as g


def init(data, state):
    # system
    state["expId"] = 'visualization_0'
    state["gpus"] = '0'
    state["confThres"] = 0.4

    # stepper
    state["collapsed4"] = True
    state["disabled4"] = True
    data["done4"] = False


def restart(data, state):
    data["done4"] = False


@g.my_app.callback("apply_parameters")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def apply_parameters(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.done4", "payload": True},
        {"field": "state.collapsed5", "payload": False},
        {"field": "state.disabled5", "payload": False},
        {"field": "state.activeStep", "payload": 5},
    ]
    api.app.set_field(task_id, "data.scrollIntoView", f"step{5}")
    g.api.app.set_fields(g.task_id, fields)
