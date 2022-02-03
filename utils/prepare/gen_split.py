from typing import Any, Callable, List, Tuple, Union
from datasets.odometry_dataset_loader import OdometryDatasetLoader

def gen_subsequences(dataset_loader, seq_id, interval = None):
    # type: (OdometryDatasetLoader, Any, int) -> List[Tuple[Any, ...]]
    seq_length = dataset_loader.seq_length
    subseqs = [] # type: List[Tuple[int, ...]]
    for subseq in dataset_loader.yield_subsequences(seq_id, interval=interval):
        row = (subseq.seq_id,) + tuple(subseq.subseq_frames_id)
        subseqs.append(row)
    return subseqs
    
def gen_split(dataset_dir, filename, dataset_loader, interval):
    sequences = dataset_loader.sequences 
    dataset = [] # type: List[str]
    for seq_id in sequences:
        subseqs = gen_subsequences(dataset_loader, seq_id, interval)
        for subseq in subseqs:
            dataset.append(' '.join(tuple(map(str, subseq))))

    dataset_txt = '\n'.join(dataset)

    with open(filename, 'w') as f:
        f.write(dataset_txt)