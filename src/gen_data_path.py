import os
import glob
import _init_paths


def gen_data_path(root_path):
    for folder_type in ['train', 'test']:
        mot_path = f'images/{folder_type}'
        real_path = os.path.join(root_path, mot_path)
        seq_names = [s for s in sorted(os.listdir(real_path))]
        with open(f'./data/sly_mot.{folder_type}', 'w') as f:
            for seq_name in seq_names:
                seq_path = os.path.join(real_path, seq_name)
                seq_path = os.path.join(seq_path, 'img1')
                images = sorted(glob.glob(seq_path + '/*.jpg'))
                len_all = len(images)
                len_half = int(len_all / 2)
                for i in range(len_all):
                    image = images[i]
                    print(image, file=f)




if __name__ == '__main__':
    root = '/FairMOT/data'
    gen_data_path(root)
