import glob
import warnings
warnings.filterwarnings('ignore')
import argparse
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
# from torchpack import distributed as dist
# from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from tqdm import tqdm

from core import builder
from core.trainers import SemanticKITTITrainer

import os

from torchpack import distributed as dist
from torchpack.utils import fs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/vacekpa2/spvnas/configs/semantic_kitti/spvcnn/augmented_default.yaml', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', default='/mnt/personal/vacekpa2/run_spvcnn', metavar='DIR', help='run directory')
    args, opts = parser.parse_known_args()


    configs.load(args.config, recursive=True)
    # configs.load('configs/semantic_kitti/spvcnn/default.yaml', recursive=True)
    configs.update(opts)

    # todo, augmented split, there are too much data

    configs.dataset.root = "/mnt/personal/vacekpa2/data/semantic_kitti/dataset/sequences/"
    exp_dir = "/mnt/personal/vacekpa2/data/semantic_kitti/dataset/spvnas/"
    weights = torch.tensor(np.load('/home/vacekpa2/spvnas/clz_weights.npy'))
    # if os.path.exists(exp_dir):
        # exp_dir += '_'

    # exp_dir += str(len(glob.glob(exp_dir + '*')))
    # os.makedirs(exp_dir, exist_ok=True)
    # exp_dir += f'-{args.run_dir}'


    if configs.distributed:
        dist.init()

    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(dist.local_rank())
    torch.cuda.set_device(0)

    def get_run_dir() -> str:
        global _run_dir
        return _run_dir

    def set_run_dir(dirpath: str) -> None:
        global _run_dir
        _run_dir = fs.normpath(dirpath)
        fs.makedir(_run_dir)

        prefix = '{time}'
        if dist.size() > 1:
            prefix += '_{:04d}'.format(dist.rank())
        # logger.add(os.path.join(_run_dir, 'logging', prefix + '.log'),
        #            format=('{time:YYYY-MM-DD HH:mm:ss.SSS} | '
        #                    '{name}:{function}:{line} | '
        #                    '{level} | {message}'))

    def auto_set_run_dir() -> str:
        tags = ['run']

        if configs:
            tags.append(configs.hash()[:8])
        run_dir = os.path.join('runs', '-'.join(tags))
        set_run_dir(run_dir)
        return run_dir

    # if args.run_dir is None:
    #     args.run_dir = auto_set_run_dir()
    # else:
    #     set_run_dir(args.run_dir)


    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2 ** 32 - 1)

    seed = configs.train.seed + dist.rank(
    ) * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = builder.make_dataset()
    dataflow = {}
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)


    model = builder.make_model().cuda()
    if configs.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.local_rank()], find_unused_parameters=True)

    # criterion = builder.make_criterion()
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(torch.float).cuda(), ignore_index=255)
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)



    trainer = SemanticKITTITrainer(exp_dir=exp_dir,
                                   model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   num_workers=configs.workers_per_gpu,
                                   seed=seed,
                                   amp_enabled=configs.amp_enabled)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    for epoch in range(configs.num_epochs):
        model.train()
        trainer.train(dataflow=dataflow['train'], num_epochs=1)

        model.eval()

        for feed_dict in tqdm(dataflow['test'], desc='eval'):
            _inputs = {}
            for key, value in feed_dict.items():
                if 'name' not in key:
                    _inputs[key] = value.cuda()

            inputs = _inputs['lidar']
            targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
            outputs = model(inputs)


            invs = feed_dict['inverse_map']

            all_labels = feed_dict['targets_mapped']

            _outputs = []
            _targets = []

            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs.append(outputs_mapped)
                _targets.append(targets_mapped)
            outputs = torch.cat(_outputs, 0)
            targets = torch.cat(_targets, 0)
            output_dict = {'outputs': outputs, 'targets': targets}


        print(f"Epoch {epoch:03d} \t"
              f"Running Loss {trainer.running_loss:.2f} \t"
              f"Mean Loss:{(trainer.running_loss / len(dataflow['train'])):.2f}")

        trainer.running_loss = 0

        states = trainer._state_dict()
        torch.save(states, f"{exp_dir}/models/{epoch:03d}.pth")

    sys.stdout.close()
if __name__ == '__main__':
    main()
