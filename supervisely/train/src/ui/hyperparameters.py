import os
import supervisely_lib as sly
import sly_globals as g


def init(data, state):
    # system
    state["expId"] = 'output_exp_name'
    state["gpusId"] = '0'
    state["workersPerGPU"] = 2

    # model
    state["headConv"] = -1

    # train
    state["epochs"] = 5
    state["lr"] = 1e-4
    state["lrStep"] = 20
    state["batchSize"] = 5
    state["masterBatchSize"] = 5
    state["valInterval"] = 1

    # loss
    state["hmWeight"] = 1
    state["offWeight"] = 0.5
    state["whWeight"] = 0.1
    state["idWeight"] = 1
    state["reidDim"] = 128
    state["ltrb"] = True

    # stepper
    state["collapsed7"] = True
    state["disabled7"] = True
    state["done7"] = False


def restart(data, state):
    data["done7"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.done4", "payload": True},
        {"field": "state.collapsed5", "payload": False},
        {"field": "state.disabled5", "payload": False},
        {"field": "state.activeStep", "payload": 5},
    ]
    g.api.app.set_fields(g.task_id, fields)
