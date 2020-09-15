# -*- coding: utf-8 -*-
""" Misc tools"""

import tensorflow.compat.v1 as tf
import numpy as np
from functools import wraps

def get_hvd():
    try:
        import horovod.tensorflow as hvd
    except ImportError as e:
        raise type(e)('Horovod package with tensoflow support must be installed to run parallel training').with_traceback(sys.exc_info()[2])
 
    try:
        hvd.size()
    except ValueError:
        hvd.init()
    return hvd

def parallelize_model(model_dir, params, kwargs):
    hvd = get_hvd()
    
    model_dir = model_dir if hvd.rank() == 0 else None
    
    model_params = params['model_params']
    
    model_params['learning_rate'] *= hvd.size()
    
    if 'training_hooks' not in model_params: model_params['training_hooks'] = []
    model_params['training_hooks'].append( hvd.BroadcastGlobalVariablesHook(0) )
    
    if 'optimizer' not in model_params: model_params['optimizer'] = tf.train.AdamOptimizer(model_params['learning_rate'])
    model_params['optimizer'] = hvd.DistributedOptimizer(model_params['optimizer'])
    
    if 'config' not in kwargs:
        kwargs['config'] = tf.estimator.RunConfig()

    kwargs['config'].session_config.gpu_options.visible_device_list = str(hvd.local_rank())
    
    params['model_params'] = model_params
    
    return model_dir, params, kwargs


def get_atomic_dress(dataset, elems, max_iter=None):
    """Fit the atomic energy with a element dependent atomic dress

    Args:
        dataset: dataset to fit
        elems: a list of element numbers
    Returns:
        atomic_dress: a dictionary comprising the atomic energy of each element
        error: residue error of the atomic dress
    """
    tensors = dataset.make_one_shot_iterator().get_next()
    if 'ind_1' not in tensors:
        tensors['ind_1'] = tf.expand_dims(tf.zeros_like(tensors['elems']), 1)
        tensors['e_data'] = tf.expand_dims(tensors['e_data'], 0)
    count = tf.equal(tf.expand_dims(
        tensors['elems'], 1), tf.expand_dims(elems, 0))
    count = tf.cast(count, tf.int32)
    count = tf.segment_sum(count, tensors['ind_1'][:, 0])
    sess = tf.Session()
    x, y = [], []
    it = 0
    while True:
        it += 1
        if max_iter is not None and it > max_iter:
            break
        try:
            x_i, y_i = sess.run((count, tensors['e_data']))
            x.append(x_i)
            y.append(y_i)
        except tf.errors.OutOfRangeError:
            break
    x, y = np.concatenate(x, 0), np.concatenate(y, 0)
    beta = np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T), np.array(y))
    dress = {e: float(beta[i]) for (i, e) in enumerate(elems)}
    error = np.dot(x, beta) - y
    return dress, error


def pi_named(default_name='unnamed'):
    """Decorate a layer to have a name"""
    def decorator(func):
        @wraps(func)
        def named_layer(*args, name=default_name, **kwargs):
            with tf.variable_scope(name):
                return func(*args, **kwargs)
        return named_layer
    return decorator


def TuneTrainable(train_fn):
    """Helper function for geting a trainable to use with Tune

    The function expectes a train_fn function which takes a config as input,
    and returns four items.

    - model: the tensorflow estimator.
    - train_spec: training specification.
    - eval_spec: evaluation specification.
    - reporter: a function which returns the metrics given evalution.

    The resulting trainable reports the metrics when checkpoints are saved,
    the report frequency is controlled by the checkpoint frequency,
    and the metrics are determined by reporter.
    """
    import os
    from ray.tune import Trainable
    from tensorflow.compat.v1.train import CheckpointSaverListener

    class _tuneStoper(CheckpointSaverListener):
        def after_save(self, session, global_step_value):
            return True

    class TuneTrainable(Trainable):
        def _setup(self, config):
            tf.logging.set_verbosity(tf.logging.ERROR)
            self.config = config
            model, train_spec, eval_spec, reporter = train_fn(config)
            self.model = model
            self.train_spec = train_spec
            self.eval_spec = eval_spec
            self.reporter = reporter

        def _train(self):
            import warnings
            index_warning = 'Converting sparse IndexedSlices'
            warnings.filterwarnings('ignore', index_warning)
            model = self.model
            model.train(input_fn=self.train_spec.input_fn,
                        max_steps=self.train_spec.max_steps,
                        hooks=self.train_spec.hooks,
                        saving_listeners=[_tuneStoper()])
            eval_out = model.evaluate(input_fn=self.eval_spec.input_fn,
                                      steps=self.eval_spec.steps,
                                      hooks=self.eval_spec.hooks)
            metrics = self.reporter(eval_out)
            return metrics

        def _save(self, checkpoint_dir):
            latest_checkpoint = self.model.latest_checkpoint()
            chkpath = os.path.join(checkpoint_dir, 'path.txt')
            with open(chkpath, 'w') as f:
                f.write(latest_checkpoint)
            return chkpath

        def _restore(self, checkpoint_path):
            with open(checkpoint_path) as f:
                chkpath = f.readline().strip()
            self.model, _, _, _ = train_fn(self.config, chkpath)
    return TuneTrainable


def connect_dist_grad(tensors):
    """This function assumes tensors is a dictionary containing 'ind_2',
    'diff' and 'dist' from a neighbor list layer It rewirtes the
    'dist' and 'dist' tensor so that their gradients are properly
    propogated during force calculations
    """
    tensors['diff'] = _connect_diff_grad(tensors['coord'], tensors['diff'],
                                         tensors['ind_2'])
    if 'dist' in tensors:
        # dist can be deleted if the jacobian is cached, so we may skip this
        tensors['dist'] = _connect_dist_grad(tensors['diff'], tensors['dist'])


def preprocess_dataset_neighbor_list(tensors, rc=5.0):
    """Generates a function that can be ``Dataset.map``ped to precompute neighbor list.

    The tensors dictionary is modified in-place. ``rc`` is the radius
    cutoff in angstrom.

    Examples
    --------
    Dataset.map(preprocess_dataset_neighbor_list)

    """
    from pinn.layers import cell_list_nl

    # This enables to precompute neighbor list even before calling sparse_batch().
    if 'ind_1' not in tensors:
        # Create a shallow copy of tensors that we can modify before feeding it to cell_list_nl.
        tensors_copy = tensors.copy()

        # Create a fake 'ind_1' index so that we can use the cell_list_nl layer.
        # 'elems' has shape (n_atoms,) while 'ind_1' needs to be (n_atoms, 1).
        tensors_copy['ind_1'] = tf.expand_dims(tf.zeros_like(tensors_copy['elems']), axis=-1)

        if 'cell' in tensors:
            # Add a fake batch dimension. 'cell' is not touched by sparsify
            # so its batch dimension correspond to the number of frames.
            tensors_copy['cell'] = tf.expand_dims(tensors_copy['cell'], 0)
    else:
        tensors_copy = tensors

    # Compute the neighbor list.
    result = cell_list_nl(tensors_copy, rc=rc)

    # Update the original dictionary.
    tensors.update(result)
    return tensors


@tf.custom_gradient
def _connect_diff_grad(coord, diff, ind):
    """Returns a new diff with its gradients connected to coord"""
    def _grad(ddiff, coord, diff, ind):
        natoms = tf.shape(coord)[0]
        if type(ddiff) == tf.IndexedSlices:
            # handle sparse gradient inputs
            ind = tf.gather_nd(ind, tf.expand_dims(ddiff.indices, 1))
            ddiff = ddiff.values
        dcoord = tf.unsorted_segment_sum(ddiff, ind[:, 1], natoms)
        dcoord -= tf.unsorted_segment_sum(ddiff, ind[:, 0], natoms)
        return dcoord, None, None
    return tf.identity(diff), lambda ddiff: _grad(ddiff, coord, diff, ind)


@tf.custom_gradient
def _connect_dist_grad(diff, dist):
    """Returns a new dist with its gradients connected to diff"""
    def _grad(ddist, diff, dist):
        return tf.expand_dims(ddist/dist, 1)*diff, None
    return tf.identity(dist), lambda ddist: _grad(ddist, diff, dist)


@pi_named('form_basis_jacob')
def make_basis_jacob(basis, diff):
    jacob = [tf.gradients(basis[:, i], diff)[0]
             for i in range(basis.shape[1])]
    jacob = tf.stack(jacob, axis=2)
    return jacob


def connect_basis_jacob(tensors):
    tensors['basis'] = _connect_basis_grad(
        tensors['diff'], tensors['basis'], tensors['jacob'])


@tf.custom_gradient
def _connect_basis_grad(diff, basis, jacob):
    def _grad(dbasis, jacob):
        ddiff = jacob * tf.expand_dims(dbasis, 1)
        ddiff = tf.reduce_sum(ddiff, axis=2)
        return ddiff, None, None
    return tf.identity(basis), lambda dbasis: _grad(dbasis, jacob)
