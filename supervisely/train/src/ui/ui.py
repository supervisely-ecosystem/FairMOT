import supervisely as sly
import sly_globals as g
import input_data as input_project


import architectures as model_architectures
import hyperparameters as hyperparameters
import splits as splits

import select_class as select_class

import monitoring as monitoring
# import artifacts as artifacts


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None

    input_project.init(data, state)  # 1 stage
    select_class.init(data, state)  # 2 stage
    splits.init(data, state)  # 3 stage
    model_architectures.init(data, state)  # 4 stage
    hyperparameters.init(data, state)  # 5 stage

    monitoring.init(data, state)  # 6 stage
    # artifacts.init(data)


@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    restart_from_step = state["restartFrom"]
    data = {}
    state = {}

    if restart_from_step <= 2:
        if restart_from_step == 2:
            select_class.restart(data, state)
        else:
            select_class.init(data, state)
    if restart_from_step <= 3:
        if restart_from_step == 3:
            splits.restart(data, state)
        else:
            splits.init(data, state)
    if restart_from_step <= 4:
        if restart_from_step == 4:
            model_architectures.restart(data, state)
        else:
            model_architectures.init(data, state)
    if restart_from_step <= 5:
        if restart_from_step == 5:
            hyperparameters.restart(data, state)
        else:
            hyperparameters.init(data, state)
    if restart_from_step <= 6:
        if restart_from_step == 6:
            monitoring.restart(data, state)
        else:
            monitoring.init(data, state)


    fields = [
        {"field": "data", "payload": data, "append": True, "recursive": False},
        {"field": "state", "payload": state, "append": True, "recursive": False},
        {"field": "state.restartFrom", "payload": None},
        {"field": f"state.collapsed{restart_from_step}", "payload": False},
        {"field": f"state.disabled{restart_from_step}", "payload": False},
        {"field": "state.activeStep", "payload": restart_from_step},
    ]
    g.api.app.set_fields(g.task_id, fields)
    g.api.app.set_field(task_id, "data.scrollIntoView", f"step{restart_from_step}")
