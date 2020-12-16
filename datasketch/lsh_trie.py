import pickle
import struct

from datasketch.lsh import _integration, _false_positive_probability, _false_negative_probability, _optimal_param

class MinHashLSHTrie(object):
    '''
    The :ref:`minhash_lsh` index.
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `Chapter 3, Mining of Massive Datasets
    <http://www.mmds.org/>`_.

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized MinHash LSH will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int, optional): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        weights (tuple, optional): Used to adjust the relative importance of
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of
            :code:`(false_positive_weight, false_negative_weight)`.
        params (tuple, optional): The LSH parameters (i.e., number of bands and size
            of each bands). This is used to bypass the parameter optimization
            step in the constructor. `threshold` and `weights` will be ignored
            if this is given.
        storage_config (dict, optional): Type of storage service to use for storing
            hashtables and keys.
            `basename` is an optional property whose value will be used as the prefix to
            stored keys. If this is not set, a random string will be generated instead. If you
            set this, you will be responsible for ensuring there are no key collisions.
        prepickle (bool, optional): If True, all keys are pickled to bytes before
            insertion. If None, a default value is chosen based on the
            `storage_config`.
        hashfunc (function, optional): If a hash function is provided it will be used to
            compress the index keys to reduce the memory footprint. This could cause a higher
            false positive rate.

    Note:
        `weights` must sum to 1.0, and the format is
        (false positive weight, false negative weight).
        For example, if minimizing false negative (or maintaining high recall) is more
        important, assign more weight toward false negative: weights=(0.4, 0.6).
        Try to live with a small difference between weights (i.e. < 0.5).
    '''

    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5, 0.5),
                 params=None):

        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError("The product of b and r in params is "
                        "{} * {} = {} -- it must be less than num_perm {}. "
                        "Did you forget to specify num_perm?".format(
                            self.b, self.r, self.b*self.r, num_perm))
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(threshold, num_perm,
                    false_positive_weight, false_negative_weight)

        self.hashranges = [(i*self.r, (i+1)*self.r) for i in range(self.b)]
        self.root = dict()
        self.keys = set()

    def insert(self, key, minhash, check_duplication=True):
        '''
        Insert a key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.

        :param str key: The identifier of the set.
        :param datasketch.MinHash minhash: The MinHash of the set.
        :param bool check_duplication: To avoid duplicate keys in the storage (`default=True`).
                                       It's recommended to not change the default, but
                                       if you want to avoid the overhead during insert
                                       you can set `check_duplication = False`.
        '''
        assert len(minhash) == self.h, ValueError("Expecting minhash with length %d, got %d"
                    % (self.h, len(minhash)))
        if check_duplication:
            assert key not in self.keys, ValueError("The given key already exists")
        
        paths = [tuple([i, ] + minhash.hashvalues[start:end].tolist())
                    for i, (start, end) in enumerate(self.hashranges)]
        for path in paths:
            cur = self.root
            leaf = path[-1]
            for node in path[:-1]:
                if node not in cur:
                    cur[node] = dict()
                    cur = cur[node]
                else:
                    cur = cur[node]
            if leaf in cur:
                cur[leaf].append(key)
            else:
                cur[leaf] = [key,]
        self.keys.add(key)

    def query(self, minhash):
        '''
        Giving the MinHash of the query set, retrieve
        the keys that references sets with Jaccard
        similarities greater than the threshold.

        Args:
            minhash (datasketch.MinHash): The MinHash of the query set.

        Returns:
            `list` of unique keys.
        '''
        assert len(minhash) == self.h, ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))

        paths = [tuple([i, ] + minhash.hashvalues[start:end].tolist())
                    for i, (start, end) in enumerate(self.hashranges)]

        candidates = set()
        for path in paths:
            cur = self.root
            leaf = path[-1]
            miss = False
            for node in path[:-1]:
                if node in cur:
                    cur = cur[node]
                else:
                    miss = True
                    break
            if miss or (leaf not in cur):
                continue
            else:
                candidates.update(cur[leaf])
        return list(candidates)

    def __contains__(self, key):
        '''
        Args:
            key (hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.
        '''
        return key in self.keys