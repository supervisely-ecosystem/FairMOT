import os
import supervisely as sly
import sly_globals as g


def init(data, state):
    # system
    state["expId"] = str(g.task_id)
    state["gpus"] = '0'
    state["numWorkers"] = 2

    # model
    state["headConv"] = -1

    # train
    state["numEpochs"] = 50
    state["lr"] = 1e-4
    state["lrStep"] = 20
    state["batchSize"] = 5
    state["masterBatchSize"] = 5
    state["numIters"] = -1
    state["saveInterval"] = 5

    # validation
    state["valInterval"] = 5
    state["K"] = 500
    state["detThres"] = 0.4

    # loss
    state["hmWeight"] = 1
    state["offWeight"] = 0.5
    state["whWeight"] = 0.1
    state["idWeight"] = 1
    state["reidDim"] = 128
    state["ltrb"] = True

    # stepper
    state["collapsed5"] = True
    state["disabled5"] = True
    data["done5"] = False


def restart(data, state):
    data["done5"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.done5", "payload": True},
        {"field": "state.collapsed6", "payload": False},
        {"field": "state.disabled6", "payload": False},
        {"field": "state.activeStep", "payload": 6},
    ]
    api.app.set_field(task_id, "data.scrollIntoView", f"step{6}")
    g.api.app.set_fields(g.task_id, fields)
