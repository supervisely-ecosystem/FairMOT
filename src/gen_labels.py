import os.path as osp
import os
import numpy as np
import json


def get_index_by_label(used_labels, label):
    return list(used_labels.keys())[list(used_labels.values()).index(label)]


def cache_label_name(labels_dict, label_name):
    cached_keys = sorted(list(labels_dict.keys()))

    if len(cached_keys) == 0:
        labels_dict[0] = label_name
    else:
        if label_name not in list(labels_dict.values()):
            last_key = max(cached_keys)
            fore_key = last_key + 1
            labels_dict[fore_key] = label_name
    return 0


def gen_labels(ds_root):
    used_labels = {}

    # ds_root = '/FairMOT/data/SLY_MOT/'
    for folder_type in ['train', 'test']:
        seq_root = osp.join(ds_root, f'images/{folder_type}')
        label_root = osp.join(ds_root, f'labels_with_ids/{folder_type}')

        os.makedirs(label_root, exist_ok=True)

        seqs = [s for s in os.listdir(seq_root)]

        tid_curr = 0
        tid_last = -1
        for seq in seqs:
            seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
            gt_names = sorted(os.listdir(osp.join(seq_root, seq, 'gt')))

            for label_index_curr, gt_name in enumerate(gt_names):
                label_name = gt_name.split('_')[1].split('.')[0]
                cache_label_name(used_labels, label_name)

                gt_txt = osp.join(seq_root, seq, 'gt', f'{gt_name}')
                gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
                idx = np.lexsort(gt.T[:2, :])
                gt = gt[idx, :]

                seq_label_root = osp.join(label_root, seq, 'img1')
                os.makedirs(seq_label_root, exist_ok=True)

                for fid, tid, x, y, w, h, mark, _, _, _ in gt:
                    if mark == 0:
                        continue
                    fid = int(fid)
                    tid = int(tid)
                    if not tid == tid_last:
                        tid_curr += 1
                        tid_last = tid
                    x += w / 2
                    y += h / 2
                    label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
                    label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        get_index_by_label(used_labels, label_name),
                        tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
                    with open(label_fpath, 'a') as f:
                        f.write(label_str)




        with open(f'{osp.join(ds_root, "classes_mapping.json")}', 'w') as fp:
            json.dump(used_labels, fp)

        print('done generating for classes:\n'
              f'{used_labels}')
