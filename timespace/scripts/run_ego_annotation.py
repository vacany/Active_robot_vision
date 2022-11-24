from timespace.egomotion import Ego_Spread
from pat.toy_dataset import Sequence_Loader

if __name__ == "__main__":
    for sequence in range(22):
        dataset = Sequence_Loader(dataset_name='semantic_kitti', sequence=sequence)
        Ego_seq = Ego_Spread(dataset=dataset)
        Ego_seq.run_spread()
