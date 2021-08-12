import supervisely_lib as sly
import sly_globals as g
import input_data as input_project


import load_models
import choose_models
import parameters
import visualize


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None

    input_project.init(data, state)  # 1 stage
    load_models.init(data, state)  # 2 stage
    choose_models.init(data, state)  # 3 stage
    parameters.init(data, state)  # 4 stage
    visualize.init(data, state)  # 5 stage



@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    restart_from_step = state["restartFrom"]
    data = {}
    state = {}

    if restart_from_step <= 2:
        if restart_from_step == 2:
            load_models.restart(data, state)
        else:
            load_models.init(data, state)

    if restart_from_step <= 3:
        if restart_from_step == 3:
            choose_models.restart(data, state)
        else:
            choose_models.init(data, state)

    if restart_from_step <= 4:
        if restart_from_step == 4:
            parameters.restart(data, state)
        else:
            parameters.init(data, state)
    if restart_from_step <= 5:
        if restart_from_step == 5:
            visualize.restart(data, state)
        else:
            visualize.init(data, state)

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
