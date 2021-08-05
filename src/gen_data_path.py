import os
import glob
import _init_paths


def gen_data_path(root_path):
    for folder_type in ['train', 'test']:

        images_mot_path = os.path.join(root_path, f'images/{folder_type}')
        labels_mot_path = os.path.join(root_path, f'labels_with_ids/{folder_type}')

        seq_names = [s for s in sorted(os.listdir(images_mot_path))]
        with open(f'./data/sly_mot.{folder_type}', 'w') as f:
            for seq_name in seq_names:
                seq_label_root = os.path.join(labels_mot_path, seq_name, 'img1')
                seq_path = os.path.join(images_mot_path, seq_name, 'img1')

                images = sorted(glob.glob(seq_path + '/*.jpg'))
                gt_files = sorted(glob.glob(seq_label_root + '/*.txt'))

                images = [image_path for image_path in images
                          if image_path.replace('images', 'labels_with_ids').replace('.jpg', '.txt') in gt_files]

                len_all = len(images)
                len_half = int(len_all / 2)
                for i in range(len_all):
                    image = images[i]
                    print(image, file=f)


if __name__ == '__main__':
    root = '/FairMOT/data'
    gen_data_path(root)
