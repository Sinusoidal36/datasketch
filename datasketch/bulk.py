from datasketch.minhash import MinHash
from collections.abc import Iterable

def compute_minhash(b, m):
    return MinHash(
            permutations=m.permutations,
            hashvalues=m._compute_hashvalues(
                b,
                permutations=m.permutations,
                hashvalues=m.hashvalues
            ),
            num_perm=m.num_perm,
            seed=m.seed,
            hashfunc=m.hashfunc
        )

def compute_minhashes(b, m):
    '''Helper method to compute minhashes in bulk. This helper avoids unnecessary
    overhead when initializing many minhashes by reusing initial state.

    Args:
        b (List[List[bytes]]): List containing lists of bytes
        m (MinHash): Initialized MinHash object
            The configuration of this MinHash will be used for all minhashes
    '''
    return list(compute_minhashes_generator(b, m))

def compute_minhashes_generator(b, m):
    '''Helper method to compute minhashes in bulk. This helper avoids unnecessary
    overhead when initializing many minhashes by reusing initial state. This method
    returns a generator for streaming computation.

    Args:
        b (iterable): Iterable containing lists of bytes
        m (MinHash): Initialized MinHash object
            The configuration of this MinHash will be used for all minhashes
    '''
    m = MinHash() if m is None else m
    assert isinstance(b, Iterable), TypeError(f'Expecting iterable, given: {type(b)}')
    for _b in b:
        yield compute_minhash(_b, m)