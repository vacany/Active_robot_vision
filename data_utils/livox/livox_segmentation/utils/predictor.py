import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torchsparse import SparseTensor
from scipy.spatial import cKDTree as KDTree


class Predictor:
    def __init__(
        self,
        net,
        loader_test,
        path_to_ckpt,
        output_dir,
        device,
        colormap=None,
    ):

        # Dataloaders
        self.loader_test = loader_test

        # Network
        self.dev = device
        self.net = net.to(self.dev)

        # Checkpoint
        self.path_to_ckpt = path_to_ckpt
        self.load_state()

        # Folder where to save the result
        self.out_dir = output_dir

        # Colormap for visualisation
        self.colormap = colormap

    @torch.no_grad()
    def eval(self):

        # Set eval mode
        net = self.net.eval()
        loader = self.loader_test
        print("\nSegment")

        # Loop over mini-batches
        bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
        for it, batch in enumerate(tqdm(loader, bar_format=bar_format)):

            # Inputs
            coords = batch["coords"]
            feats = batch["feats"]
            # Get logits on quantized point cloud
            net_inputs = SparseTensor(
                coords=coords.to(self.dev, non_blocking=True),
                feats=feats.to(self.dev, non_blocking=True)
            )
            with torch.autocast(self.dev, enabled=True):
                out = net(net_inputs)
                out = out[batch["closest_point"]]
            out = out.float()

            # Get final prediction (log_prob and class index)
            log_prob = torch.nn.functional.log_softmax(out, 1)
            max_log_prob, class_index = log_prob.max(1, keepdim=True)

            # Save result
            for ind, filename in enumerate(batch["filenames"]):
                filename = self.out_dir + "/" + filename
                os.makedirs(os.path.split(filename)[0], exist_ok=True)
                where = coords[batch["closest_point"], -1] == ind
                if self.colormap is None:
                    prediction = np.concatenate(
                        (
                            batch["pc_list"][ind],
                            class_index[where].cpu().numpy(),
                            torch.exp(max_log_prob[where]).cpu().numpy(),
                        ),
                        axis=1
                    )
                    np.savez(filename[:-4] + "_seg.npz", prediction=prediction)
                else:
                    prediction = np.concatenate(
                        (
                            batch["pc_list"][ind],
                            self.colormap[class_index[where, 0].cpu().numpy()],
                        ),
                        axis=1
                    )
                    np.savetxt(
                        filename[:-3] + "txt",
                        prediction,
                    )

    def load_state(self):
        ckpt = torch.load(self.path_to_ckpt, map_location=torch.device(self.dev))
        pattern = list(ckpt["net"].keys())[0].split(".")[0]
        if pattern == "module" or pattern == "net":
            state_dict = {}
            for key in ckpt["net"].keys():
                new_key = key[len(pattern + "."):]
                state_dict[new_key] = ckpt["net"][key]
        else:
            state_dict = ckpt["net"]
        self.net.load_state_dict(state_dict)
