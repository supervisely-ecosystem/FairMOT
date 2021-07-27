<div align="center" markdown>
<img src="https://i.imgur.com/ufxMDIH.png"/>

# Serve MMClassification

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#For-Developers">For Developers</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmclassification/supervisely/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmclassification)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmclassification/supervisely/serve&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmclassification/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmclassification/supervisely/serve&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

App deploys MMClassification model trained in Supervisely as REST API service. Serve app is the simplest way how any model 
can be integrated into Supervisely. Once model is deployed, user gets the following benefits:

1. Use out of the box apps for inference - [AI assisted classification and tagging](https://ecosystem.supervise.ly/apps/ai-assisted-classification)
2. Apps from Supervisely Ecosystem can use NN predictions: for visualization, for analysis, performance evaluation, etc ...
3. Communicate with NN in custom python script (see section <a href="#For-developers">for developers</a>)
4. App illustrates how to use NN weights. For example: you can train model in Supervisely, download its weights and use 
   them the way you want outside Supervisely.

Watch usage demo:

<a data-key="sly-embeded-video-link" href="https://youtu.be/HwIgu_f4duU" data-video-code="HwIgu_f4duU">
    <img src="https://i.imgur.com/tohTu5R.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:70%;">
</a>


# How To Run

1. Go to the directory with weights in `Team Files`. Training app saves results to the 
   directory: `/mmclassification/<session id>_<training project name>/checkpoints`. Then right click to weights `.pth` file,
   for example: `/mmclassification/6181_synthetic products v2/checkpoints/latest.pth`
   
<img src="https://i.imgur.com/cmEzYGr.gif"/>

2. Run `Serve MMClassification` app from context menu

3. Select device, both `gpu` and `cpu` are supported. Also in advanced section you can 
change what agent should be used for deploy.

4. Press `Run` button.

5. Wait until you see following message in logs: `Model has been successfully deployed`

<img src="https://i.imgur.com/AAKToCb.png" width="800"/>

6. All deployed models are listed in `Team Apps`. You can view logs and stop them from this page.

<img src="https://i.imgur.com/7eVkiIm.png"/>


# For Developers

This python example illustrates available methods of the deployed model. Now you can integrate network predictions to 
your python script. This is the way how other Supervisely Apps can communicate with NNs. And also you can use serving 
app as an example - how to use downloaded NN weights outside Supervisely.


## Python Example: how to communicate with deployed model 
```python
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
```

## Example Output

Information about deployed model:

```json
{
    "app": "MM Classification Serve",
    "weights": "/mmclassification/777_animals/checkpoints/best_accuracy_top-1_epoch_44.pth",
    "device": "cuda:0",
    "session_id": 6918,
    "classes_count": 3
}
```

Model produces following tags:
```
ProjectMeta:

Tags
+------------+------------+-----------------+--------+---------------+--------------------+
|    Name    | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
+------------+------------+-----------------+--------+---------------+--------------------+
| cat        |    none    |       None      |        |      all      |         []         |
+------------+------------+-----------------+--------+---------------+--------------------+
| dog        |    none    |       None      |        |      all      |         []         |
+------------+------------+-----------------+--------+---------------+--------------------+
| fox        |    none    |       None      |        |      all      |         []         |
+------------+------------+-----------------+--------+---------------+--------------------+
```

Image examples (urls) for predicted tags:
```json
{
   "cat": [
      "https://cfa.org/wp-content/uploads/2019/11/abtProfile.jpg",
      "https://cfa.org/wp-content/uploads/2019/11/abyProfile.jpg"
   ],
   "dog": [
      "https://i.imgur.com/R2bI8hi.jpg",
      "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/26155623/Siberian-Husky-standing-outdoors-in-the-winter.jpg"
   ],
   "fox": [
      "https://cdn.britannica.com/30/145630-050-D1B34751/Red-foxes-insects-rodents-fruit-grain-carrion.jpg",
      "https://cdn.britannica.com/35/174035-050-E2AB419D/red-fox-hill.jpg"
   ]
}
```

Predictions example (predictions are sorted by score):
```json
[
    {
        "label": 1,
        "score": 0.89,
        "class": "dog"
    },
    {
        "label": 0,
        "score": 0.08,
        "class": "cat"
    }
]
```
