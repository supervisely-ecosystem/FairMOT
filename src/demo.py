from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import glob


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    models_paths = sorted(glob.glob(os.path.join(opt.load_model, '*.pth')))
    # models_paths = [models_paths[-1]]
    # print(models_paths)
    for model_path in models_paths:
        print(model_path)
        opt.load_model = model_path
        for index, video_path in enumerate(sorted(glob.glob(
                os.path.join(opt.input_video, '*')))):

            exp_name = opt.load_model.split('/')[-2]
            e = model_path.split('/')[-1].split('_')[-1][:-4]

            logger.info(f'e: {e}')
            logger.info(f'exp: {exp_name}')
            logger.info('Starting tracking...')
            logger.info(f'vp: {video_path}')
            dataloader = datasets.LoadVideo(video_path, opt.img_size)

            frame_rate = dataloader.frame_rate

            # frame_dir = None if opt.output_format == 'text' else osp.join(result_root, exp_name, e, f'frame_{index}')
            frame_dir = osp.join(result_root, exp_name, e, f'frame_{index}')
            result_filename = osp.join(result_root, exp_name, e, f'results_{index}.txt')

            os.makedirs(frame_dir, exist_ok=True)

            eval_seq(opt, dataloader, 'mot', result_filename,
                     save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
                     use_cuda=opt.gpus!=[-1])

            if opt.output_format == 'video':

                output_video_path = osp.join(result_root, exp_name, e)
                os.makedirs(output_video_path, exist_ok=True)
                output_video_path = osp.join(result_root, exp_name, e, f'results_{index}.mp4')
                cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(frame_dir, output_video_path)
                os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
