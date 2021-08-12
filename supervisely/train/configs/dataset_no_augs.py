# dataset settings
dataset_type = 'Supervisely'

input_size = 256
batch_size_per_gpu = 32
num_workers_per_gpu = 2
validation_interval = 1
save_best = None
project_dir = None

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(input_size, input_size)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(input_size, input_size)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=batch_size_per_gpu,
    workers_per_gpu=num_workers_per_gpu,
    train=dict(
        type=dataset_type,
        project_dir=project_dir,
        data_prefix='train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        project_dir=project_dir,
        data_prefix='val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        project_dir=project_dir,
        data_prefix='test',
        pipeline=test_pipeline),
)
evaluation = dict(interval=validation_interval, save_best=save_best, metric='accuracy')
