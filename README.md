# Bucketized Sequences for Keras

This package is intended to speed up training of variable-length sequences in Keras by defining a Sequence object (dataset) that puts sequences of similar lengths in buckets.

The inputs of this class (found in `bucketed_sequence.py`) are the number of buckets (ranges are dynamically generated using a histogram), ideal minibatch size (sometimes smaller minibatches will be produced due to small bucket sizes), the sequences, output values, and their respective lengths.

A usage example can be found in `test.py`, which trains an LSTM to return the third value in a given sequence. If run with `--buckets=0`, the code will run normally, without the bucketed sequence.

License: New BSD (3-clause).