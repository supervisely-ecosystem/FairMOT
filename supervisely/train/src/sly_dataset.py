import os
import copy
import numpy as np
import supervisely_lib as sly

from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS


@DATASETS.register_module()
class Supervisely(BaseDataset):
    """`Supervisely <https://supervise.ly/>`_ Dataset.
    """

    CLASSES = None

    def __init__(self, project_dir, data_prefix, pipeline, test_mode=False):
        self.gt_labels = sly.json.load_json_file(os.path.join(project_dir, "gt_labels.json"))
        Supervisely.CLASSES = sorted(self.gt_labels, key=self.gt_labels.get)
        self.split_name = data_prefix
        self.items = sly.json.load_json_file(os.path.join(project_dir, "splits.json"))[self.split_name]
        self.project_fs = sly.Project(project_dir, sly.OpenMode.READ)
        super(Supervisely, self).__init__(data_prefix=self.split_name, pipeline=pipeline, test_mode=test_mode)

    def load_annotations(self):
        classes_set = set(Supervisely.CLASSES)
        data_infos = []
        for paths in self.items:
            img_path = paths["img_path"]
            ann_path = paths["ann_path"]
            if not sly.fs.file_exists(img_path):
                sly.logger.warn(f"File {img_path} not found and item will be skipped")
                continue
            if not sly.fs.file_exists(ann_path):
                sly.logger.warn(f"File {ann_path} not found and item will be skipped")
                continue
            ann = sly.Annotation.load_json_file(ann_path, self.project_fs.meta)
            img_tags = {tag.name for tag in ann.img_tags}

            gt_label = list(img_tags.intersection(classes_set))
            if len(gt_label) != 1:
                sly.logger.warn(f"File {ann_path} has {len(gt_label)} gt labels")
                continue
            else:
                gt_label = gt_label[0]

            gt_index = self.gt_labels[gt_label]
            data_infos.append({
                "img_prefix": self.split_name,
                "img_info": {'filename': img_path},
                "gt_label": np.array(gt_index, dtype=np.int64)
            })
        return data_infos
