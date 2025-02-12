import os
import torch
import pickle
import numpy as np
from definitions import ROOT_DIR

from data.tu_utils import load_data, S2V_to_PyG, get_fold_indices
from data.utils import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings
from data.datasets import InMemoryComplexDataset


class TUDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from TUDatasets."""

    def __init__(self, root, name, max_dim=2, num_classes=2, degree_as_tag=False,
                 init_method='sum', seed=0, include_down_adj=False, max_ring_size=None):
        self.name = name
        self.degree_as_tag = degree_as_tag
        assert max_ring_size is None or max_ring_size >= 3
        self._max_ring_size = max_ring_size
        cellular = (max_ring_size is not None)
        if cellular:
            assert max_dim == 2

        super(TUDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            init_method=init_method, include_down_adj=include_down_adj, cellular=cellular)

        # Load processed data (now the entire dataset is loaded)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Remove the K-fold related code: No fold, train_ids, val_ids
        self.test_ids = None  # No test ids as well

    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(TUDataset, self).processed_dir
        suffix = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix += f"_down_adj" if self.include_down_adj else ""
        return directory + suffix

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pt'.format(self.name)]

    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG.
        return ['{}_graph_list_degree_as_tag_{}.pkl'.format(self.name, self.degree_as_tag)]

    def download(self):
        # This will process the raw data into a list of PyG Data objs.
        data, num_classes = load_data(self.raw_dir, self.name, self.degree_as_tag)
        self._num_classes = num_classes
        print('Converting graph data into PyG format...')
        graph_list = [S2V_to_PyG(datum) for datum in data]
        with open(self.raw_paths[0], 'wb') as handle:
            pickle.dump(graph_list, handle)

    def process(self):
        with open(self.raw_paths[0], 'rb') as handle:
            graph_list = pickle.load(handle)

        if self._cellular:
            print("Converting the dataset accounting for rings...")
            complexes, _, _ = convert_graph_dataset_with_rings(graph_list, max_ring_size=self._max_ring_size,
                                                               include_down_adj=self.include_down_adj,
                                                               init_method=self._init_method,
                                                               init_edges=True, init_rings=True)
        else:
            print("Converting the dataset with gudhi...")
            complexes, _, _ = convert_graph_dataset_with_gudhi(graph_list, expansion_dim=self.max_dim,
                                                               include_down_adj=self.include_down_adj,
                                                               init_method=self._init_method)

        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])

    def get_tune_idx_split(self):
        raise NotImplementedError('Not implemented yet')
