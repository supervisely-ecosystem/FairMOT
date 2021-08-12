from shutil import copytree
import os


exp_dir = '../exp/mot'
output_logs_dir = '../logs_dir/'


print(os.listdir(exp_dir))

for curr_exp in os.listdir(exp_dir):
    curr_path = os.path.join(exp_dir, curr_exp)
    logs_name = [name for name in os.listdir(curr_path) if name.startswith('logs')][0]
    print(logs_name)

    dst = os.path.join(output_logs_dir, f"{curr_exp}/")
    # os.makedirs(dst, exist_ok=True)

    copytree(os.path.join(curr_path, f"{logs_name}/"), dst)


# os.path.join()
#
# copyfile(src, dst)