import tensorflow as tf
import sys
import os

BATCH_SIZE=1024

def input_fn(batch_size=BATCH_SIZE):
    feature_description = {
        'price': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'category': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'elder': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'shaper': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'corrupted': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'num_sockets': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'bool_mods': tf.io.VarLenFeature(tf.string),
        'bool_mods_source': tf.io.VarLenFeature(tf.string),
        'scaled_mods_name': tf.io.VarLenFeature(tf.string),
        'scaled_mods_value': tf.io.VarLenFeature(tf.float32),
        'scaled_mods_source': tf.io.VarLenFeature(tf.string),
        }

    def _PopLabel(features):
        label = features.pop('price')
        return features, label
    return (
        tf.data.TFRecordDataset('examples.rio')
        .map(lambda x: tf.io.parse_single_example(x, feature_description))
        .map(_PopLabel)
        .shuffle(30000)
        .repeat()
        .batch(batch_size)
        )    


def BuildModEmbeddings(names, values, source):        
    NUM_NODES_EMBEDDING = [3, 3]
    # Maximum number of mods per item.
    MAX_MODS = 15
    # Number of different mods in total. +1 for empty mod
    NUM_MODS = 1185 +1 +1
    
    scaled_mod_table = tf.lookup.StaticVocabularyTable(tf.lookup.TextFileInitializer(
        'scaled_mods-dict.txt', tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), num_oov_buckets=1)
    sources_table = tf.lookup.StaticVocabularyTable(tf.lookup.KeyValueTensorInitializer(
        keys=['implicitMods', 'explicitMods', 'enchantMods', 'craftedMods'],
        values=[x for x in range(4)],
        key_dtype=tf.string, value_dtype=tf.int64), num_oov_buckets=1)

    mod_ids = scaled_mod_table.lookup(names)
    mod_ids = tf.sparse.reset_shape(mod_ids,
                                    [BATCH_SIZE, MAX_MODS])
    mod_ids = tf.sparse.to_dense(mod_ids, default_value=MAX_MODS - 1)
    values = tf.sparse.reset_shape(values, [BATCH_SIZE, MAX_MODS])
    values = tf.sparse.expand_dims(values, -1)
    values  = tf.sparse.to_dense(values)
    
    source = sources_table.lookup(source)
    source = tf.sparse.reset_shape(source, [BATCH_SIZE, MAX_MODS])
    source = tf.sparse.expand_dims(source, -1)
    source = tf.sparse.to_indicator(source, 5)  # Num sources
    source = tf.cast(source, tf.float32)

    mods_raw = tf.concat([values, source], 2)
    mods_raw = tf.expand_dims(mods_raw, 2)
    
    indices = tf.reshape(mod_ids, [-1])
    all_weights = tf.constant([0.0])
    for i, num_nodes in enumerate(NUM_NODES_EMBEDDING):
        o_shape = mods_raw.shape
        kernel = tf.get_variable('mod_embedding_kernel_%d' % i,
                                 [NUM_MODS, o_shape[-1], num_nodes])
        bias = tf.get_variable('mod_embedding_bias_%d' % i,
                               [NUM_MODS, num_nodes])
        k = tf.reshape(tf.gather(kernel, indices),
                       [-1, o_shape[1], o_shape[-1], num_nodes])
        b = tf.reshape(tf.gather(bias, indices),
                       [-1, o_shape[1], 1, num_nodes])
        all_weights += (tf.reduce_sum(tf.abs(tf.reshape(k, [-1]))) +
                        tf.reduce_sum(tf.abs(tf.reshape(b, [-1])))
                        )
                        
        mods_raw = tf.nn.relu(tf.matmul(mods_raw, k) + b)
    
    mods_raw = tf.squeeze(mods_raw, axis=2)

    result = tf.math.reduce_sum(mods_raw, 1)
    return result, all_weights
    

def model(features, labels, mode):
    LAYER_CONFIG = [40, 20]
    feature_columns = [
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity('elder', 2)),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity('shaper', 2)),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity('corrupted', 2)),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity('num_sockets', 7)),
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_file('bool_mods', 'bool_mods-dict.txt'),
            10),
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_file('name', 'names-dict.txt'),
            10),
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_file('category', 'categories-dict.txt'),
            10),
        ]
    simple_inputs = tf.feature_column.input_layer(features, feature_columns)
    mod_embeddings, all_weights = BuildModEmbeddings(features['scaled_mods_name'],
                       features['scaled_mods_value'],
                       features['scaled_mods_source'])
    hidden_layer = tf.concat([simple_inputs, mod_embeddings], axis=-1)

    for num_nodes in LAYER_CONFIG:
        hidden_layer = tf.layers.dense(hidden_layer, num_nodes, tf.nn.relu)

    prediction = tf.layers.dense(hidden_layer, 1, tf.nn.relu)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, {'price': tf.exp(prediction)})
    log_labels = tf.log(labels)
    loss = (tf.reduce_mean((prediction - log_labels) ** 2)
            + 1e-2 * tf.reduce_mean((tf.exp(prediction) - labels)**2)
            + 1e-3 * all_weights
            )
    tf.summary.scalar('price/mean-square-error-log', tf.reduce_mean((prediction - log_labels) ** 2))
    tf.summary.scalar('price/mean-square-error', tf.reduce_mean((tf.exp(prediction) - labels) ** 2))
    tf.summary.scalar('price/predicted', tf.reduce_mean(tf.exp(prediction)))
    tf.summary.scalar('price/log_predicted', tf.reduce_mean(prediction))
    tf.summary.scalar('price/real', tf.reduce_mean(labels))
    tf.summary.scalar('price/log_real', tf.reduce_mean(log_labels))
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    optimizer = tf.train.AdagradOptimizer(learning_rate=3e-2)
    #train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    grads, variables = zip(*optimizer.compute_gradients(loss))
    grads,  _ = tf.clip_by_global_norm(grads, 5.0)
    train_op = optimizer.apply_gradients(zip(grads, variables),
                                         global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

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
        tf.feature_column.categorical_column_with_vocabulary_file('category', 'categories-dict.txt'),
        ]
    
    #estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
    estimator = tf.estimator.Estimator(model_fn=model, model_dir='./model')
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=500000)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=5000)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    main()
