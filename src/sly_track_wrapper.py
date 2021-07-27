from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import sly_eval_seq

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress



def main(opt):
    exp_name_folder = opt.exp_id
    models_rows = g.api.app.get_field(g.task_id, 'data.modelsTable')
    selected_models = g.api.app.get_field(g.task_id, 'state.selectedModels')

    models_progress = get_progress_cb('Models', "Current checkpoint", len(selected_models), min_report_percent=1)

    for row in models_rows:
        if not (row['name'] in selected_models):
            continue

        model_epoch = row['epoch']
        model_arch = row['arch']
        model_name = row['name']

        opt.load_model = os.path.join(g.checkpoints_dir, model_name)
        opt.arch = model_arch

        output_root = os.path.join(g.output_dir, exp_name_folder, f'{model_name.replace(".", "_")}')

        videos_data = g.api.app.get_field(g.task_id, 'data.videosData')

        data_type = 'mot'

        # run tracking
        accs = []
        n_frame = 0
        timer_avgs, timer_calls = [], []

        videos_progress = get_progress_cb('Videos', "Processing video",
                                          len(videos_data), min_report_percent=1)

        for index, video_data in enumerate(videos_data):
            video_path = video_data['path']
            frame_rate = video_data['fps']
            video_index = video_data['index']

            video_name = video_path.split('/')[-1]

            dataloader = datasets.LoadImages(video_path, opt.img_size)

            result_filename = os.path.join(output_root, 'tracks', f'{video_index}.txt')
            os.makedirs(os.path.join(output_root, 'tracks'), exist_ok=True)

            nf, ta, tc = sly_eval_seq.eval_seq(opt, dataloader, data_type, result_filename,
                                               save_dir=output_root, show_image=False, frame_rate=frame_rate,
                                               epoch=model_epoch)

            n_frame += nf
            timer_avgs.append(ta)
            timer_calls.append(tc)

            # eval
            logger.info(f'Evaluate seq: {video_index}')
            evaluator = Evaluator(g.converted_dir, video_name, data_type)
            accs.append(evaluator.eval_file(result_filename))

            if opt.output_format == 'video':
                output_video_path = osp.join(output_root, 'videos', f'{video_index}.mp4')
                os.makedirs(osp.join(output_root, 'videos'), exist_ok=True)

                cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v libx264 {}'.format(output_root, output_video_path)
                os.system(cmd_str)

                for file in os.listdir(output_root):
                    if file.endswith('.jpg'):
                        os.remove(os.path.join(output_root, file))
            videos_progress(1)

        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

        models_progress(1)

    reset_progress('Models')
    reset_progress('Videos')


if __name__ == '__main__':
    opt = opts().init()
    main(opt)
