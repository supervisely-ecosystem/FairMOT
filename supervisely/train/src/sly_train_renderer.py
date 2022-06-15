import sly_globals as g
import os

from supervisely_lib.app.widgets import CompareGallery

from datetime import timedelta


def update_charts(e, data):
    charts_data = preprocess_train_metrics(e, data)
    send_fields(charts_data, need_append=True)


def preprocess_train_metrics(e, data):
    charts_data = {}

    def to_camel_case(snake_str):
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)

    for item, value in data.items():
        charts_data[f"data.chart{to_camel_case(item)}.series[0].data"] = \
            [[float(e), float(value.avg if hasattr(value, 'avg') else value)]]

    return charts_data


def send_fields(fields_and_values, need_append=False):
    fields = []
    for field, value in fields_and_values.items():
        fields.append(
            {"field": field, "payload": value, "append": need_append}
        )

    g.api.app.set_fields(g.task_id, fields)


def preview_predictions(gt_image, pred_image):
    gallery_preview = CompareGallery(g.task_id, g.api, f"data.galleryPreview", g.project_meta)
    append_gallery(gt_image, pred_image)

    update_preview_by_index(-1, gallery_preview)


def update_preview_by_index(index, gallery_preview):
    previews_links = g.api.app.get_field(g.task_id, 'data.previewPredLinks')
    detection_threshold = g.api.app.get_field(g.task_id, 'state.detThres')
    gt_image_link = previews_links[index][0]
    pred_image_link = previews_links[index][1]

    gallery_preview.set_left('ground truth', gt_image_link)
    gallery_preview.set_right(f'predicted [threshold: {detection_threshold}]',
                              pred_image_link)

    gallery_preview.update(options=False)


def save_and_upload_image(temp_image, img_type):
    remote_preview_path = "/temp/{}_preview_detections.jpg"
    local_image_path = os.path.join(g.my_app.data_dir, f"{img_type}.jpg")
    g.sly.image.write(local_image_path, temp_image)
    if g.api.file.exists(g.team_id, remote_preview_path.format(img_type)):
        g.api.file.remove(g.team_id, remote_preview_path.format(img_type))

    # @TODO: add ann in SLY format
    # class_lemon = g.sly.ObjClass('lemon', g.sly.Rectangle)
    # label_lemon = g.sly.Label(g.sly.Rectangle(200, 200, 500, 600), class_lemon)
    #
    # labels_arr = [label_lemon]
    # height, width = temp_image.shape[0], temp_image.shape[1]
    # ann = g.sly.Annotation((height, width), labels_arr)

    file_info = g.api.file.upload(g.team_id, local_image_path, remote_preview_path.format(img_type))
    return file_info


def append_gallery(gt_image, pred_image):
    file_info_gt = save_and_upload_image(gt_image, 'gt')
    file_info_pred = save_and_upload_image(pred_image, 'pred')

    fields = [
        {"field": "data.previewPredLinks",
         "payload": [[file_info_gt.storage_path, file_info_pred.storage_path]], "append": True},
    ]

    g.api.app.set_fields(g.task_id, fields)

    fields = [
        {"field": "state.currEpochPreview",
         "payload": len(g.api.app.get_field(g.task_id, 'data.previewPredLinks')) *
                    g.api.app.get_field(g.task_id, 'state.valInterval')},
    ]

    g.api.app.set_fields(g.task_id, fields)


def calculate_eta_epochs(eta_iter_value, eta_iter_elapsed_value, curr_epoch):
    each_epoch_times = g.api.app.get_field(g.task_id, 'data.etaEpochData')
    epochs_num = g.api.app.get_field(g.task_id, 'state.numEpochs')

    if curr_epoch == 1:
        each_epoch_times.append(eta_iter_value + eta_iter_elapsed_value)

    mean_epoch_time = sum(each_epoch_times) / len(each_epoch_times)

    eta_epoch_seconds = mean_epoch_time * (epochs_num - curr_epoch + 1) - eta_iter_elapsed_value
    return "{:0>8}".format(str(timedelta(seconds=round(eta_epoch_seconds))))


def finish_training_in_advance():
    return g.api.app.get_field(g.task_id, 'state.finishTrain')
