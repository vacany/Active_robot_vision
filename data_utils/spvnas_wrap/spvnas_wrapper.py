import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import torch.cuda
import torch.nn as nn
import yaml

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

from data_utils.basics import Basic_Dataprocessor




def get_device_idx_for_port():  # TODO not for multiple gpus
    gpu_txt = open('/home/vacekpa2/gpu.txt', 'r').readlines()
    os.system('nvidia-smi -L > /home/vacekpa2/gpu_all.txt')
    import time
    time.sleep(0.1)
    gpu_all_txt = open('/home/vacekpa2/gpu_all.txt', 'r').readlines()

    gpu_all_txt = [text[7:] for text in gpu_all_txt]
    device_idx = 0
    for idx, gpu_id in enumerate(gpu_all_txt):
        if gpu_txt[0][7:] == gpu_id:
            device_idx = idx

    return device_idx

def make_model(model_name, num_classes, voxel_size) -> nn.Module:
    cr = 1.0    # crop from config
    if model_name == 'minkunet':
        from data_utils.spvnas_wrap.core.models.semantic_kitti import MinkUNet
        model = MinkUNet(num_classes=num_classes, cr=cr)

    elif model_name == 'spvcnn':
        from data_utils.spvnas_wrap.core.models.semantic_kitti import SPVCNN

        model = SPVCNN(num_classes=num_classes,
                       cr=cr,
                       pres=voxel_size,
                       vres=voxel_size)
    else:
        raise NotImplementedError(model_name + " not implemented, choose between [spvcnn, minkunet]")
    return model



# preprocessor?
class SPVCNN_Preprocesor(torch.utils.data.Dataset):
    '''
    !!! NOT FASTER, num workers overhead is slowing it down. Implementation failed.
    Multithreaded preprocessing of data. Module created to separate process of preparing raw data for each model differently
    '''
    def __init__(self, batch_size, voxel_size=0.05):

        self.batch_size = batch_size
        self.voxel_size = voxel_size

        self.loader = torch.utils.data.DataLoader(self,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=sparse_collate_fn)
    def __call__(self, raw_data, num_points):
        self.data_dict = raw_data
        self.num_points = num_points

        return next(iter(self.loader))


    def __getitem__(self, index):
        block = self.data_dict[index]['pts'][:, :4].copy()
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        labels_ = self.data_dict[index]['label_mapped']
        # if 'train' in split:  #TODO is it important for training speed etc.?
        if len(inds) > self.num_points:
            inds = np.random.choice(inds, self.num_points, replace=False)

        else:
            inds = torch.arange(0, self.num_points)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        data = {
                'pts': lidar,
                'used_pts_idx': inds,
                'target': labels,
                'label_mapped': labels_,
                'inverse_map': inverse_map,
                'filename': self.data_dict[index]['filename']
        }

        return data

    def __len__(self):
        return self.batch_size

def preprocess_batch(batch, voxel_size, num_points=80000):
    '''

    :param batch: {pts: , labels:}
    :param num_points:
    :param voxel_size:
    :return:
    '''
    inputs = []

    for data_dict in batch:
        block = data_dict['pts'][:, :4].copy()
        pc_ = np.round(block[:, :3] / voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        labels_ = data_dict['label_mapped']
        # if 'train' in split:  #TODO is it important for training speed etc.?
        if len(inds) > num_points:
            inds = torch.tensor(np.random.choice(inds, num_points, replace=False))

        else:
            inds = torch.arange(0, num_points)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        inputs.append({
                'pts': lidar,
                'used_pts_idx' : inds,  # torch tensor
                'target': labels,
                'label_mapped': labels_,
                'inverse_map': inverse_map,
                'filename': data_dict['filename']
        })

    return sparse_collate_fn(inputs)

if __name__ == '__main__':
    # TODO Refaktor this to importable functions
    BATCH_SIZE = 8
    DATASET_NAME = 'synlidar'
    IGNORE_CLASS = 255 if DATASET_NAME == 'semantic_kitti' else 0
    # You need config ...
    dataset = Basic_Dataprocessor(dataset_name=DATASET_NAME, sequence=4)
    data_loader = dataset.get_data_loader(dataset, batch_size=BATCH_SIZE)
    val_loader = dataset.get_data_loader(dataset, batch_size=1)

    with open(os.path.dirname(os.path.abspath(__file__)) + '/spvnas_config.yaml') as f:
        default_cfg = yaml.load(f, Loader=yaml.Loader)

    model = make_model(default_cfg['model'], default_cfg['num_classes'] + 1, default_cfg['voxel_size'])
    # set gpu
    device_idx = get_device_idx_for_port()
    torch.cuda.set_device(device_idx)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_CLASS)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    # model_weights = torch.load('/home/vacekpa2/toy_set/spvcnn.pt', map_location='cpu')
    # model.load_state_dict(model_weights)
    model = model.cuda()
    model.train()

    for epoch in range(2):
        for idx, raw_data in enumerate(data_loader):

            batch = preprocess_batch(raw_data, voxel_size=0.05, num_points=80000)   # takes long time, to optimize on threads?

            _inputs = {}
            for key, value in batch.items():
                if 'name' not in key:
                    _inputs[key] = value.cuda()

            inputs = _inputs['pts'].cuda()
            outputs, decoder_features = model(inputs, return_features=True)

            targets = batch['target'].F.long().cuda()#(non_blocking=True)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch: {epoch:03d} \t Iter: {idx} \t Loss: {loss.item()}")



        with torch.no_grad():
            model.eval()
            for idx, val_data in enumerate(val_loader):

                pts_size = len(val_data[0]['pts'])
                batch = preprocess_batch(val_data, voxel_size=0.05, num_points=pts_size)

                _inputs = {}
                for key, value in batch.items():
                    if 'name' not in key:
                        _inputs[key] = value.cuda()

                inputs = _inputs['pts'].cuda()
                invs = _inputs['inverse_map']
                all_labels = _inputs['label_mapped']

                outputs = model(inputs, return_features=False)


                output_dict = {'pts' : val_data[0]['pts'],
                               'output': outputs.argmax(1).detach().cpu().numpy(),
                               'target': all_labels.F.detach().cpu().numpy()}
                # torch.save(model.state_dict(), '/home/vacekpa2/toy_set/spvcnn.pt')
            # save orig_pts, orig_preds

                # os.makedirs('/home/vacekpa2/toy_set/spvcnn_out/', exist_ok=True)
                # for key, values in output_dict.items():
                #     np.save(f"/home/vacekpa2/toy_set/spvcnn_out/{key}_{idx:06d}.npy", output_dict[key])

                # gather indices from part of the training samples (80000 points only)
                # for idx in range(invs.C[:, -1].max() + 1):
                #     orig_output = np.zeros(raw_data[idx]['pts'].shape[0])
                #     orig_target = raw_data[idx]['labels_mapped']
                #
                #     cur_scene_pts = (inputs.C[:, -1] == idx).cpu()
                #     pts_indices = batch['used_pts_idx'][idx]
                #
                #     outputs_frame = outputs[cur_scene_pts].argmax(1)
                #     orig_output[pts_indices] = outputs_frame.detach().cpu()
                    # _outputs.append(orig_output)
                    # _targets.append(orig_target)
                #
                # output_dict = {'pts': [raw_data[i]['pts'] for i in range(len(raw_data))], 'outputs': _outputs, 'targets': _targets}
                # # torch.save(model.state_dict(), '/home/vacekpa2/toy_set/spvcnn.pt')
                # # save orig_pts, orig_preds
                #
                # os.makedirs('/home/vacekpa2/toy_set/spvcnn_out/', exist_ok=True)
                # for key, values in output_dict.items():
                # for i in range(len(values)):
                #     np.save(f"/home/vacekpa2/toy_set/spvcnn_out/{key}_{i:06d}.npy", output_dict[key][i])
