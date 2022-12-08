def make_sequence(dataset_name, sequence_nbr):

    if 'delft' in dataset_name:
        from data_utils.delft_dataset import Delft_Sequence
        Sequence = Delft_Sequence(sequence_nbr)

    elif 'argoverse' in dataset_name:
        pass

    else:
        raise NotImplementedError("Dataset not found")

    return Sequence



# REserved for class dataloader
# def collate_fn(self, inputs):
#     return [i for i in inputs]
#
# @staticmethod
# def get_data_loader(self, shuffle=False, batch_size=4):
#     return torch.utils.data.DataLoader(self,
#                                        batch_size=batch_size,
#                                        num_workers=batch_size,
#                                        shuffle=shuffle,
#                                        collate_fn=self.collate_fn)
# return {
#                 'pts': pts,
#                 'global_pts': global_pts,
#                 # 'orig_label': all_labels,
#                 'pose': self.poses[index],
#                 'label_mapped': labels_,
#                 'instance': instance,
#                 'filename': self.pts_files[index]
#         }

