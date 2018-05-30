import math
import random
import numpy as np
from keras import utils


def _roundto(val, batch_size):
    return int(math.ceil(val / batch_size)) * batch_size

    
class BucketedSequence(utils.Sequence):
    """
    A Keras Sequence (dataset reader) of input sequences read in bucketed bins.
    Assumes all inputs are already padded using `pad_sequences` (where padding 
    is prepended).
    """

    def __init__(self, num_buckets, batch_size, seq_lengths, x_seq, y):
        self.batch_size = batch_size
        # Count bucket sizes
        bucket_sizes, bucket_ranges = np.histogram(seq_lengths,
                                                   bins=num_buckets)
        
        # Obtain the (non-sequence) shapes of the inputs and outputs
        input_shape = (1,) if len(x_seq.shape) == 2 else x_seq.shape[2:]
        output_shape = (1,) if len(y.shape) == 1 else y.shape[1:]
        
        # Looking for non-empty buckets
        actual_buckets = [bucket_ranges[i+1] 
                          for i,bs in enumerate(bucket_sizes) if bs > 0]
        actual_bucketsizes = [bs for bs in bucket_sizes if bs > 0]
        bucket_seqlen = [int(math.ceil(bs)) for bs in actual_buckets]
        num_actual = len(actual_buckets)
        print('Training with %d non-empty buckets' % num_actual)
        #print(bucket_seqlen)
        #print(actual_bucketsizes)
        self.bins = [(np.ndarray([bs, bsl] + list(input_shape), dtype=x_seq.dtype),
                      np.ndarray([bs] + list(output_shape), dtype=x_seq.dtype)) 
                     for bsl,bs in zip(bucket_seqlen, actual_bucketsizes)]
        assert len(self.bins) == num_actual

        # Insert the sequences into the bins
        bctr = [0]*num_actual
        for i,sl in enumerate(seq_lengths):
            for j in range(num_actual):
                bsl = bucket_seqlen[j]
                if sl < bsl or j == num_actual - 1:
                    self.bins[j][0][bctr[j],:bsl] = x_seq[i,-bsl:]
                    self.bins[j][1][bctr[j],:] = y[i]
                    bctr[j] += 1
                    break

        self.num_samples = x_seq.shape[0]
        self.dataset_len = int(sum([math.ceil(bs / self.batch_size) 
                                    for bs in actual_bucketsizes]))
        self._permute()            

    def _permute(self):
        # Shuffle bins
        random.shuffle(self.bins)

        # Shuffle bin contents
        for i, (xbin, ybin) in enumerate(self.bins):
            index_array = np.random.permutation(xbin.shape[0])
            self.bins[i] = (xbin[index_array], ybin[index_array])

    def on_epoch_end(self):
        self._permute()

    def __len__(self):
        """ Returns the number of minibatches in this sequence. """
        return self.dataset_len

    def __getitem__(self, idx):
        idx_begin, idx_end = self.batch_size*idx, self.batch_size*(idx+1)

        # Obtain bin index
        for i,(xbin,ybin) in enumerate(self.bins):
            rounded_bin = _roundto(xbin.shape[0], self.batch_size)
            if idx_begin >= rounded_bin:
                idx_begin -= rounded_bin
                idx_end -= rounded_bin
                continue
                
            # Found bin
            idx_end = min(xbin.shape[0], idx_end) # Clamp to end of bin
            
            return xbin[idx_begin:idx_end], ybin[idx_begin:idx_end]

            
        raise ValueError('out of bounds')
