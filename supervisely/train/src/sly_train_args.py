import sys
import sly_globals as g
# import train_config
import re


def init_script_arguments(state):
    sys.argv = []

    def camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    needed_args = ['expId', 'gpusId', 'workersPerGPU', 'epochs', 'lr', 'lrStep',
                   'batchSize', 'masterBatchSize', 'valInterval', 'hmWeight', 'offWeight',
                   'whWeight', 'idWeight', 'reidDim', 'ltrb']

    for needed_arg in needed_args:
        sys.argv.extend([f'--{camel_to_snake(needed_arg)}', state[needed_arg]])

    sys.argv.extend([f'--arch', state["selectedModel"]])
