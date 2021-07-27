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


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def add_stats_to_image(image, opt):
    model_name = opt.load_model.split('/')[-1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{model_name}"

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    # textX = int((image.shape[1] - textsize[0]))
    # textY = int(textsize[1] * 2)

    textX = int((image.shape[1] - textsize[0]) / 2)
    textY = int((image.shape[0] + textsize[1]) / 2)

    # add text centered on image
    cv2.putText(image, text, (textX, textY), font, 1, (0, 0, 255), 2)

    return image


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True,
             epoch=0):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    # for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        # if i % 8 != 0:
        # continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        # online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            # vertical = tlwh[2] / tlwh[3] > 1.6
            # if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
            if tlwh[2] * tlwh[3] > opt.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                # online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        # results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
            online_im = add_stats_to_image(online_im, opt)

        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    # write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


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

            nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
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
