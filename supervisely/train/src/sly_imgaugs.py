from mmcls.datasets.builder import PIPELINES
import supervisely_lib as sly


@PIPELINES.register_module()
class SlyImgAugs(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = sly.json.load_json_file(self.config_path)
        self.augs = sly.imgaug_utils.build_pipeline(self.config["pipeline"], random_order=self.config["random_order"])

    def _apply_augs(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            res_img = sly.imgaug_utils.apply_to_image(self.augs, img)
            results[key] = res_img
            results['img_shape'] = res_img.shape

    def __call__(self, results):
        self._apply_augs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(config_path={self.config_path})'
        return repr_str