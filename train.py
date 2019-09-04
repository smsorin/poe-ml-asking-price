import tensorflow as tf

def input_fn(batch_size=128):
    feature_description = {
        'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'price': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'elder': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'shaper': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'corrupted': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'num_sockets': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'bool_mods': tf.io.VarLenFeature(tf.string),
        'scaled_mods_name': tf.io.VarLenFeature(tf.string),
        'scaled_mods_value': tf.io.VarLenFeature(tf.float32),
        }

    def _PopLabel(features):
        label = features.pop('price')
        return features, label
    return (
        tf.data.TFRecordDataset('examples.rio')
        .map(lambda x: tf.io.parse_single_example(x, feature_description))
        .map(_PopLabel)
        .shuffle(1000)
        .repeat()
        .batch(batch_size)
        )    

def main():
    tf.logging.set_verbosity(tf.logging.INFO) 
    feature_columns = [
        tf.feature_column.categorical_column_with_identity('elder', 2),            
        tf.feature_column.categorical_column_with_identity('shaper', 2),
        tf.feature_column.categorical_column_with_identity('corrupted', 2),
        tf.feature_column.categorical_column_with_identity('num_sockets', 7),
        tf.feature_column.categorical_column_with_vocabulary_file('bool_mods', 'bool_mods-dict.txt'),
        tf.feature_column.categorical_column_with_vocabulary_file('scaled_mods_name', 'scaled_mods-dict.txt'),
        tf.feature_column.categorical_column_with_vocabulary_file('name', 'names-dict.txt'),
        ]
    
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=500000)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=5000)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
