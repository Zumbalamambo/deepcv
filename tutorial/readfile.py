import tensorflow as tf


def one_reader_one_example():
    # create a FIFO queue
    filenames = ['a.csv', 'b.csv', 'c.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

    # create reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # create decoder
    example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])

    # run graph
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.start_queue_runners(coord=coord)
        for i in range(10):
            print(example.eval())
        coord.request_stop()
        coord.join(threads)


def one_reader_multi_example():
    # create a FIFO queue
    filenames = ['a.csv', 'b.csv', 'c.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

    # create reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # create decoder
    example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])

    example_batch, label_batch = tf.train.batch([example, label], batch_size=5)

    # run graph
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            print(example_batch.eval())
        coord.request_stop()
        coord.join(threads)


def multi_reader_multi_example():
    # create a FIFO queue
    filenames = ['a.csv', 'b.csv', 'c.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

    # create reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    record_defaults = [['null'], ['null']]
    example_list = [tf.decode_csv(value, record_defaults=record_defaults) for _ in range(2)]

    example_batch, label_batch = tf.train.batch_join(example_list, batch_size=5)

    # run graph
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                print(example_batch.eval())
        except tf.errors.OutOfRangeError:
            print('epoches completed!')
        finally:
            coord.request_stop()

        coord.join(threads)


