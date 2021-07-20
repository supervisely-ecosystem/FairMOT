# checkpoint saving
checkpoint_config = dict(interval=1)

log_interval = 10
# yapf:disable
log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='SuperviselyLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
