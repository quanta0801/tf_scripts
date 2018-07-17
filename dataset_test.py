import tensorflow as tf

train_path, test_path = iris_data.maybe_download()

def csv_input_fn(train_path, batch_size):
    ds = tf.data.TextLineDataset(train_path).skip(1)

    # Metadata describing the text columns
    COLUMNS = ['SepalLength', 'SepalWidth',
            'PetalLength', 'PetalWidth',
            'label']
    FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]

    def _parse_line(line):
        # Decode the line into its fields
        fields = tf.decode_csv(line, FIELD_DEFAULTS)

        # Pack the result into a dictionary
        features = dict(zip(COLUMNS,fields))

        # Separate the label from the features
        label = features.pop('label')

        return features, label

    ds = ds.map(_parse_line)
    ds = ds.shuffle(1000).repeat().batch(batch_size)

    iterator = ds.make_one_shot_iterator()

    features, label = iterator.get_next()

    return features, label

# All the inputs are numeric
feature_columns = [
    tf.feature_column.numeric_column(name)
    for name in iris_data.CSV_COLUMN_NAMES[:-1]]

# Build the estimator
est = tf.estimator.LinearClassifier(feature_columns,
                                    n_classes=3)
# Train the estimator
batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda : csv_input_fn(train_path, batch_size))