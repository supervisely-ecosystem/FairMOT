import sys
import sly_globals as g
# import train_config
import re

import torch


def init_script_arguments(state):
    sys.argv = []

    sys.argv.extend([f'task', 'mot'])

    def camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    needed_args = ['expId', 'gpus', 'numWorkers', 'numEpochs', 'lr', 'lrStep',
                   'batchSize', 'masterBatchSize', 'valInterval', 'hmWeight', 'offWeight',
                   'whWeight', 'idWeight', 'reidDim', 'ltrb', 'saveInterval', 'detThres', 'numIters']

    for needed_arg in needed_args:
        sys.argv.extend([f'--{camel_to_snake(needed_arg)}', f'{state[needed_arg]}'])

    if state["weightsInitialization"] == "custom":
        model_path = f'{g.my_app.data_dir}/custom_model.pth'
        arch = torch.load(model_path, map_location='cpu')['arch']
        sys.argv.extend([f'--arch', arch])
        sys.argv.extend([f'--load_model', model_path])
    else:
        sys.argv.extend([f'--arch', state["selectedModel"]])

    sys.argv.extend([f'--K', str(state["K"])])
    sys.argv.extend([f'--data_cfg', f'{g.my_app.data_dir}/sly_mot_generated.json'])

