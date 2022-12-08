from robots.odometry_sequence import Ego_Spread
from data_utils.basics import Basic_Dataprocessor

if __name__ == "__main__":
    for sequence in range(22):
        dataset = Basic_Dataprocessor(dataset_name='argoverse2_train', sequence=sequence)
        Ego_seq = Ego_Spread(dataset=dataset)
        Ego_seq.run_spread()
