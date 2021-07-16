import sly_globals as g


def init(data, state):
    project_info = g.project_info
    # project_meta = g.project_meta
    project_meta = objects_project_meta = g.api.project.get_meta(id=g.project_id)

    rows_names = generate_rows_by_ann(project_meta)

    if len(rows_names) > 0:
        state['selectedClass'] = rows_names[0]['name']

    data["totalVideosCount"] = project_info.items_count

    data["selectClassTable"] = rows_names

    # state["selectClassTable"] = {
    #     "count": {
    #         "total": 0,
    #         "train": 1,
    #         "val": 2
    #     },
    #     "percent": {
    #         "total": 100,
    #         "train": 0,
    #         "val": 100
    #     },
    # }

    data["done2"] = False
    state["collapsed2"] = not True
    state["disabled2"] = not True


def restart(data, state):
    state['done2'] = False


def generate_rows_by_ann(ann_meta):
    rows = []

    for index, curr_class in enumerate(ann_meta['classes']):

        rows.append({

            "name": f"{curr_class['title']}",
            "shapeType": f"{curr_class['shape']}",
            "color": f"{curr_class['color']}",
        })

    return rows