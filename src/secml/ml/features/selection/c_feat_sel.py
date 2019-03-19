from __future__ import print_function

from secml.core import CCreator


class CFeatSel(CCreator):

    def __init__(self, criterion='standard'):
        '''
        Constructor
        '''
        self.criterion = criterion

    def selection(self, dataset, *args):
        if self.criterion == 'standard':
            return self._feat_sel(dataset)
        elif self.criterion == 'secure':
            return self._feat_sel_secure(dataset)
        elif self.criterion == 'super_secure':
            return self._feat_sel_sup_secure(dataset, *args)
        elif self.criterion == 'only_manifest':
            return self._feat_sel_only_manifest(dataset, *args)

        else:
            print(self.criterion)
            raise ValueError("Unknown criterion for feature selection.")

    def _feat_sel(self, dataset):
        score = abs(dataset.X[dataset.Y == 1, :].mean(axis=0) -
                    dataset.X[dataset.Y == 0, :].mean(axis=0))

        # TODO: this ravel should not be necessary
        score = score.ravel()

        idx = (-score).argsort()
        return idx, score

    def _feat_sel_secure(self, dataset):
        score = dataset.X[dataset.Y == 1, :].mean(axis=0) - \
                dataset.X[dataset.Y == 0, :].mean(axis=0)

        # TODO: this ravel should not be necessary
        score = score.ravel()

        idx = (-score).argsort()

        return idx, score

    #
    #     def _feat_sel_sup_secure(self, dataset,max_feat,selectionable_idx):
    #
    #
    #         score = dataset.X[dataset.Y==1,:].mean(axis=0) - \
    #                 dataset.X[dataset.Y==0,:].mean(axis=0)
    #
    #         #TODO: this ravel should not be necessary
    #         score = score.ravel()
    #
    #         all_idx = (-score).argsort() #for every pattern
    #
    #         import numpy as np
    #         selectionable_feat_idx_ordered = CArray(np.intersect1d(ar1=idx.data, ar2=selectionable_idx.data)
    #
    #         #all_idx
    #
    #         return idx, score
    #

    def _feat_sel_only_manifest(self, dataset, not_in_manifest):
        """
        not in manifest is a boolean vector with 1 for feat that aren't into manifest, zero for manifest
        """

        score = abs(dataset.X[dataset.Y == 1, :].mean(axis=0) - \
                    dataset.X[dataset.Y == 0, :].mean(axis=0))

        # TODO: this ravel should not be necessary
        score = score.ravel()

        in_manifest_feat_idx = not_in_manifest.nnz_indices[1]

        score[in_manifest_feat_idx] = 0

        idx = (-score).argsort()

        return idx, score