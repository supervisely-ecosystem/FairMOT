# by default there is no learning rate schedule
# enable it by uncommenting one of the following examples and modify its settings
# learn more in official mmcv docs:
# https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/lr_updater.html
# Note 1: `step` argument should be less than the number of epochs

# ***********************************************
# Configs examples (PLEASE, KEEP IT IN ONE LINE):
# ***********************************************

#lr_config = dict(policy='fixed')
#lr_config = dict(policy='step', step=[100, 150])
#lr_config = dict(policy='step', step=[30, 60, 90])
#lr_config = dict(policy='step', step=[40, 80, 120])
#lr_config = dict(policy='CosineAnnealing', min_lr=0)
#lr_config = dict(policy='step', gamma=0.98, step=1)
#lr_config = dict(policy='poly', min_lr=0, by_epoch=False, warmup='constant', warmup_iters=5000)
#lr_config = dict(policy='step', warmup='linear', warmup_iters=2500, warmup_ratio=0.25, step=[30, 60, 90])
#lr_config = dict(policy='CosineAnnealing', min_lr=0, warmup='linear', warmup_iters=10000, warmup_ratio=1e-4)
#lr_config = dict(policy='CosineAnnealing', min_lr=0, warmup='linear', warmup_iters=2500, warmup_ratio=0.25)
#lr_config = dict(policy='CosineAnnealing', min_lr=0, warmup='linear', warmup_iters=10000, warmup_ratio=1e-4)

