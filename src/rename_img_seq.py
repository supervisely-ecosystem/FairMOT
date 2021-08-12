import glob
import os


datasets = os.listdir('/FairMOT/data/SLY_MOT/images/train/')

for ds in datasets:
    

    paths = sorted(glob.glob(f'/FairMOT/data/SLY_MOT/images/train/{ds}/img1/*.jpg'))

    print(paths)

    for i in range(len(paths)):
        os.rename(paths[i], '{}/{:06d}_temp.jpg'.format('/'.join(paths[i].split('/')[:-1]), i + 1))


    paths = sorted(glob.glob(f'/FairMOT/data/SLY_MOT/images/train/{ds}/img1/*.jpg'))

    print(paths)

    for i in range(len(paths)):
        os.rename(paths[i], '{}/{:06d}.jpg'.format('/'.join(paths[i].split('/')[:-1]), i + 1))
