from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from absl import app, flags
import numpy as np

from bucketed_sequence import BucketedSequence

UNK = -1.0
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train')
flags.DEFINE_integer('lstm_units', 100, 'Number of LSTM units in RNN')
flags.DEFINE_integer('dense_breadth', 32, 'Number of neurons in the dense ' +
                     'layer')

flags.DEFINE_integer('dataset_size', 10000, 'Size of training dataset')
flags.DEFINE_integer('val_size', 1000, 'Size of validation set')
flags.DEFINE_integer('seqlen_mean', 50, 'Sequence length mean (drawn ' +
                     'from normal distribution)')
flags.DEFINE_integer('seqlen_stddev', 200, 'Sequence length standard ' +
                     'deviation (drawn from normal distribution)')
                     
flags.DEFINE_integer('buckets', 10, 'Number of buckets to use (run with ' +
                     '0 to disable)')

def pad(seqs, maxlen):
    # NOTE: prepends data
    padded = np.array(pad_sequences(seqs, maxlen=maxlen, value=UNK, 
                                    dtype=seqs[0].dtype))
    return np.vstack([np.expand_dims(x, axis=0) for x in padded])

def gen_dataset(set_size):
    sequence_lengths = np.random.normal(loc=FLAGS.seqlen_mean, 
        scale=FLAGS.seqlen_stddev, size=set_size).astype(np.int32)
    max_length = np.max(sequence_lengths)
    # Clamp range to start from three elements
    sequence_lengths = np.clip(sequence_lengths, 3, max_length)   
    
    # Generate random sequences
    seq_x  = [np.random.uniform(1.0, 50.0, sl) for sl in sequence_lengths]
    seq_y = np.array([seq[2] for seq in seq_x], dtype=np.float32)
    
    # Pad sequences
    padded_x = pad(seq_x, max_length)
    padded_x = np.reshape(padded_x,(len(sequence_lengths),max_length,1))
    
    # Return dataset
    return padded_x, seq_y, sequence_lengths


# Trains an LSTM to return the third (non-UNK) value in a sequence
def main(argv):
    del argv # Ignore other arguments

    # Set up a simple network (LSTM + Dense)
    inp = Input(shape=(None, 1), dtype="float32", name="in")
    lstm = LSTM(FLAGS.lstm_units, return_sequences=False,
                name="lstm")(inp)
    dense = Dense(FLAGS.dense_breadth, kernel_initializer='normal',
                  activation='relu')(lstm)
    outputs = Dense(1, kernel_initializer='normal')(dense)
    model = Model(inputs=inp, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error", 
                  metrics=['mae'])
    
    # Generate dataset
    x_train, y_train, len_train = gen_dataset(FLAGS.dataset_size)
    x_val, y_val, len_val = gen_dataset(FLAGS.val_size)
    
    
    if FLAGS.buckets > 0:
        # Create Sequence objects
        train_generator = BucketedSequence(FLAGS.buckets, FLAGS.batch_size,
                                           len_train, x_train, y_train)
        val_generator = BucketedSequence(FLAGS.buckets, FLAGS.batch_size,
                                         len_val, x_val, y_val)

        model.fit_generator(train_generator, epochs=FLAGS.epochs,
                            validation_data=val_generator,
                            shuffle=True, verbose=True)
    else:
        # No bucketing
        model.fit(x=x_train, y=y_train, epochs=FLAGS.epochs,
                  validation_data=(x_val, y_val),
                  batch_size=FLAGS.batch_size, verbose=True, shuffle=True)

if __name__ == '__main__':
    app.run(main)
