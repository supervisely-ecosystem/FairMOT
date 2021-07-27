import json
import supervisely_lib as sly


def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 6918

    # get information about model
    info = api.task.send_request(task_id, "get_session_info", data={})
    print("Information about deployed model:")
    print(json.dumps(info, indent=4))

    # get model output tags
    meta_json = api.task.send_request(task_id, "get_model_meta", data={})
    model_meta = sly.ProjectMeta.from_json(meta_json)
    print("Model predicts following tags:")
    print(model_meta)

    # get urls for tags model predicts
    # during training this information is saved and model can share this information by request
    urls_for_tags = api.task.send_request(task_id, "get_tags_examples", data={})
    print("Image examples (urls) for predicted tags:")
    print(json.dumps(urls_for_tags, indent=4))

    # get predictions by image url
    predictions = api.task.send_request(task_id, "inference_image_url", data={
        "image_url": "https://i.imgur.com/R2bI8hi.jpg",
        "topn": 2 # optional
    })
    print("Predictions for url")
    print(json.dumps(predictions, indent=4))

    # get predictions by image-id in Supervisely
    predictions = api.task.send_request(task_id, "inference_image_id", data={
        "image_id": 927270,
        "topn": 2  # optional
    })
    print("Predictions for image by id")
    print(json.dumps(predictions, indent=4))

    # get predictions for image ROI
    predictions = api.task.send_request(task_id, "inference_image_id", data={
        "image_id": 927270,
        "topn": 2,  # optional
        "rectangle": [10, 20, 150, 80]  # top, left, bottom, right
    })
    print("Predictions for image ROI")
    print(json.dumps(predictions, indent=4))


if __name__ == "__main__":
    main()