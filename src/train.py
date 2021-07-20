from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model  # SLY CODE
from models.model import save_model as save_model_base  # SLY CODE
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from test_det import test_det

from sly_train_progress import get_progress_cb, reset_progress, init_progress  # SLY CODE
import sly_train_renderer  # SLY CODE
from functools import partial  # SLY CODE





def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)

    print(os.listdir('./'))
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0


    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    save_model = partial(save_model_base, arch=opt.arch)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    sly_epoch_progress = get_progress_cb("Epoch", "Epoch", opt.num_epochs, min_report_percent=1)  # SLY CODE

    for epoch in range(1, opt.num_epochs + 1):

        if sly_train_renderer.finish_training_in_advance() and epoch != 1:  # SLY CODE
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
            break

        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_validation.pth'),
                       epoch, model, optimizer)
            with torch.no_grad():
                opt.load_model = os.path.join(opt.save_dir, 'model_validation.pth')
                mean_map, mean_r, mean_p = test_det(opt, batch_size=opt.master_batch_size)  # SLY CODE

                sly_train_renderer.update_charts(epoch - 1,
                                                 dict(zip([
                                                     "val_map",
                                                     "val_recall",
                                                     "val_precision",
                                                 ], [mean_map, mean_r, mean_p])))  # SLY CODE

            os.remove(os.path.join(opt.save_dir, 'model_validation.pth'))  # SLY CODE


        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.save_interval > 0 and epoch % opt.save_interval == 0: # SLY CODE
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                   epoch, model, optimizer)  # SLY CODE

        sly_epoch_progress(1)  # SLY CODE


    reset_progress('Epoch')  # SLY CODE
    logger.close()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().parse()
    print(opt)
    main(opt)
