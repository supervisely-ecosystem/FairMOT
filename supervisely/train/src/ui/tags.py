import os
from collections import defaultdict
import supervisely_lib as sly

import input_project
import splits
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress

tag2images = None
tag2urls = None
images_without_tags = []
disabled_tags = []

progress_index = 3
_preview_height = 120
_max_examples_count = 12

_ignore_tags = ["train", "val"]
_allowed_tag_types = [sly.TagValueType.NONE]

image_slider_options = {
    "selectable": False,
    "height": f"{_preview_height}px"
}

selected_tags = None


def init(data, state):
    data["tagsBalance"] = None
    state["selectedTags"] = []
    state["tagsInProgress"] = False
    data["tagsBalanceOptions"] = {
        "selectable": True,
        "collapsable": True,
        "clickableName": False,
        "clickableSegment": False,
        "maxHeight": "400px"
    }
    data["imageSliderOptions"] = image_slider_options
    data["done3"] = False
    data["skippedTags"] = []
    state["collapsed3"] = True
    state["disabled3"] = True
    init_progress(progress_index, data)


def restart(data, state):
    data["done3"] = False


def init_cache(split_items, split_name, progress_cb):
    global tag2images, tag2urls, images_without_tags
    for item in split_items:
        name = item.name
        dataset_name = item.dataset_name
        ann_path = item.ann_path
        img_info = input_project.get_image_info_from_cache(dataset_name, name)

        ann = sly.Annotation.load_json_file(ann_path, g.project_meta)
        if len(ann.img_tags) == 0:
            images_without_tags.append(img_info)
        else:
            for tag in ann.img_tags:
                tag2images[tag.name][split_name].append(img_info)
                tag2urls[tag.name].append({
                    "moreExamples": [img_info.full_storage_url],
                    "preview": g.api.image.preview_url(img_info.full_storage_url, height=_preview_height)
                })
        progress_cb(1)


@g.my_app.callback("show_tags")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def show_tags(api: sly.Api, task_id, context, state, app_logger):
    global tag2images, tag2urls, disabled_tags

    tag2images = defaultdict(lambda: defaultdict(list))
    tag2urls = defaultdict(list)

    progress = get_progress_cb(progress_index, "Calculate stats", g.project_info.items_count)
    init_cache(splits.train_set, "train", progress)
    init_cache(splits.val_set, "val", progress)

    segments = [
        {"name": "train", "key": "train", "color": "#13ce66"},
        {"name": "val", "key": "val", "color": "#ffa500"},
    ]

    disabled_tags = []
    _working_tags = set(tag2images.keys())
    for tag_meta in g.project_meta.tag_metas:
        if tag_meta.name not in _working_tags:
            # tags with 0 images will be ignored automatically
            disabled_tags.append({
                "name": tag_meta.name,
                "color": sly.color.rgb2hex(tag_meta.color),
                "reason": "0 images with this tag"
            })

    max_count = -1
    tags_balance_rows = []
    for tag_name, segment_infos in tag2images.items():
        tag_meta = g.project_meta.get_tag_meta(tag_name)
        tag_meta: sly.TagMeta

        if tag_name.lower() in _ignore_tags:
            disabled_tags.append({
                "name": tag_name,
                "color": sly.color.rgb2hex(tag_meta.color),
                "reason": "name is reserved"
            })
            continue

        if tag_meta.value_type not in _allowed_tag_types:
            disabled_tags.append({
                "name": tag_name,
                "color": sly.color.rgb2hex(tag_meta.color),
                "reason": "unsupported type, app supports only tags of type None (without value). Use app 'Unpack key-value tags' from Ecosystem to transform key-value tags to None tags"
            })
            continue

        train_count = len(segment_infos["train"])
        val_count = len(segment_infos["val"])

        disabled = False
        if train_count == 0:
            disabled = True
            disabled_tags.append({
                "name": tag_name,
                "color": sly.color.rgb2hex(tag_meta.color),
                "reason": "0 examples in train set, regenerate train/val splits"
            })

        total = train_count + val_count
        tags_balance_rows.append({
            "name": tag_name,
            "total": total,
            "disabled": disabled,
            "segments": {
                "train": train_count,
                "val": val_count,
            }
        })
        max_count = max(max_count, total)

    rows_sorted = sorted(tags_balance_rows, key=lambda k: k["total"], reverse=True)
    tags_balance = {
        "maxValue": max_count,
        "segments": segments,
        "rows": rows_sorted
    }

    subsample_urls = {tag_name: urls[:_max_examples_count] for tag_name, urls in tag2urls.items()}
    reset_progress(progress_index)

    fields = [
        {"field": "state.tagsInProgress", "payload": False},
        {"field": "data.tagsBalance", "payload": tags_balance},
        {"field": "data.tag2urls", "payload": subsample_urls},
        {"field": "data.skippedTags", "payload": disabled_tags}
    ]
    g.api.app.set_fields(g.task_id, fields)


@g.my_app.callback("use_tags")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_tags(api: sly.Api, task_id, context, state, app_logger):
    global selected_tags
    selected_tags = state["selectedTags"]

    fields = [
        {"field": "data.done3", "payload": True},
        {"field": "state.collapsed4", "payload": False},
        {"field": "state.disabled4", "payload": False},
        {"field": "state.activeStep", "payload": 4},
    ]
    g.api.app.set_fields(g.task_id, fields)