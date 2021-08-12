import os
import shutil

root_path = '../demo_output'

dirs = list(os.walk(root_path))

for frame_dir in [f'frame_{i}' for i in range(3)]:
    for dir in dirs:
        if dir[0].endswith(f'{frame_dir}'):
            shutil.rmtree(dir[0])



print('done')