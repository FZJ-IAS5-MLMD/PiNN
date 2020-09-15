# -*- coding: utf-8 -*-
"""Basic functions for dataset loaders"""

import random
import tensorflow.compat.v1 as tf
from pinn.utils import get_hvd

def distribute_data(*args):
    def chunkify(a, n):
        if n > len(a): raise IndexError("List of size {0} cannot be split into {1} chunks".format(len(a), n))
        j, k = len(a)//n, len(a)%n
        return [a[i * j + min(i, k):(i + 1) * j + min(i + 1, k)] for i in range(n)]
    
    chunked_data = []
    
    hvd = get_hvd()
    
    chunked_data = [chunkify( a, hvd.size() )[ hvd.rank() ] for a in args]
        
    return chunked_data

class _datalist(list):
    """The same thing as list, but don't count in nested structure
    """
    pass


def sparse_batch(batch_size, drop_remainder=False, num_parallel_calls=8,
                 atomic_props=['f_data', 'q_data', 'f_weights'], store_n_atoms=True):
    """This returns a dataset operation that transforms single samples
    into sparse batched samples. The atomic_props must include all
    properties that are defined on an atomic basis besides 'coord' and
    'elems'.

    Args:
        drop_remainder (bool): option for padded_batch
        num_parallel_calls (int): option for map
        atomic_props (list): list of atomic properties
    """

    def sparsify(tensors):
        # Sparsify atoms.
        atom_ind = tf.cast(tf.where(tensors['elems']), tf.int32)
        tensors['ind_1'] = atom_ind[:, :1]
        tensors['elems'] = tf.gather_nd(tensors['elems'], atom_ind)
        tensors['coord'] = tf.gather_nd(tensors['coord'], atom_ind)

        # Sparsify pairs.
        if 'ind_2' in tensors:
            # Collect all th non-padded indices. In padded pairs, both atom indices are zero.
            pair_ind = tf.cast(tf.where(tf.reduce_sum(tensors['ind_2'], axis=-1)), tf.int32)

            # Compute the shift in the atom indices for each pair.
            shift = tf.concat([tf.zeros(1, dtype=tensors['n_atoms'].dtype), tensors['n_atoms'][:-1]], axis=0)
            tensors['ind_2'] = tensors['ind_2'] + tf.expand_dims(tf.expand_dims(shift, -1), -1)

            # Remove padded indices.
            tensors['ind_2'] = tf.gather_nd(tensors['ind_2'], pair_ind)
            tensors['diff'] = tf.gather_nd(tensors['diff'], pair_ind)
            tensors['dist'] = tf.gather_nd(tensors['dist'], pair_ind)

        # Optional fields.
        for name in atomic_props:
            if name in tensors:
                tensors[name] = tf.gather_nd(tensors[name], atom_ind)

        return tensors

    def _store_n_atoms(item):
        item['n_atoms'] = tf.shape(item['elems'])[0]
        return item

    def _sparse_batch(dataset):
        if store_n_atoms:
            dataset = dataset.map(_store_n_atoms, num_parallel_calls)
        dataset = dataset.padded_batch(batch_size, tf.data.get_output_shapes(dataset), drop_remainder=drop_remainder)
        dataset = dataset.map(sparsify, num_parallel_calls)
        return dataset

    return _sparse_batch


def map_nested(fn, nested):
    """Map fn to the nested structure
    """
    if isinstance(nested, dict):
        return {k: map_nested(fn, v) for k, v in nested.items()}
    if isinstance(nested, list) and type(nested) != _datalist:
        return [map_nested(fn, v) for v in nested]
    else:
        return fn(nested)


def flatten_nested(nested):
    """Retun a list of the nested elements
    """
    if isinstance(nested, dict):
        return sum([flatten_nested(v) for v in nested.values()], [])
    if isinstance(nested, list) and type(nested) != _datalist:
        return sum([flatten_nested(v) for v in nested], [])
    else:
        return [nested]


def split_list(data_list, split={'train': 8, 'vali': 1, 'test': 1},
               shuffle=True, seed=None):
    """
    Split the list according to a given ratio

    Args:
        to_split (list): a list to split
        split_ratio: a nested (list and dict) of split ratio

    Returns:
        A nest structure of splitted data list
    """
    import math
    dummy = _datalist(data_list)
    if shuffle:
        random.seed(seed)
        random.shuffle(dummy)
    data_tot = len(dummy)
    split_tot = float(sum(flatten_nested(split)))

    def get_split_num(x): return math.ceil(data_tot*x/split_tot)
    split_num = map_nested(get_split_num, split)

    def _pop_data(n):
        to_pop = dummy[:n]
        del dummy[:n]
        return _datalist(to_pop)
    splitted = map_nested(_pop_data, split_num)
    return splitted


def list_loader(pbc=False, force=False, format_dict=None):
    """Decorator for building dataset loaders"""
    from functools import wraps
    if format_dict is None:
        format_dict = {
            'elems': {'dtype':  tf.int32,   'shape': [None]},
            'coord': {'dtype':  tf.float32, 'shape': [None, 3]},
            'e_data': {'dtype': tf.float32, 'shape': []},
        }
        if pbc:
            format_dict['cell'] = {'dtype':  tf.float32, 'shape': [3, 3]}
        if force:
            format_dict['f_data'] = {'dtype':  tf.float32, 'shape': [None, 3]}

    def decorator(func):
        @wraps(func)
        def data_loader(data_list, split={'train': 8, 'vali': 1, 'test': 1},
                        shuffle=True, seed=0):
            def _data_generator(data_list):
                for data in data_list:
                    yield func(data)
            dtypes = {k: v['dtype'] for k, v in format_dict.items()}
            shapes = {k: v['shape'] for k, v in format_dict.items()}

            def generator_fn(data_list): return tf.data.Dataset.from_generator(
                lambda: _data_generator(data_list), dtypes, shapes)
            subsets = split_list(data_list, split, shuffle, seed)
            splitted = map_nested(generator_fn, subsets)
            return splitted
        return data_loader
    return decorator
