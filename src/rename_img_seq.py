import glob
import os

paths = sorted(glob.glob('/FairMOT/data/SLY_MOT/images/train/MOT20-01/img1/*.jpg'))

print(paths)

for i in range(len(paths)):
    os.rename(paths[i], '{}/{:06d}_temp.jpg'.format('/'.join(paths[i].split('/')[:-1]), i + 1))


paths = sorted(glob.glob('/FairMOT/data/SLY_MOT/images/train/MOT20-01/img1/*.jpg'))

print(paths)

for i in range(len(paths)):
    os.rename(paths[i], '{}/{:06d}.jpg'.format('/'.join(paths[i].split('/')[:-1]), i + 1))
