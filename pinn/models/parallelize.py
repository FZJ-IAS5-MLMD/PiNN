import horovod.tensorflow as hvd
import tensorflow.compat.v1 as tf

def parallelize_model(model_dir, model_params, config):
    hvd.init()
    
    model_dir = model_dir if hvd.rank() == 0 else None
    
    model_params['learning_rate'] *= hvd.size()
    
    if 'training_hooks' not in model_params: model_params['training_hooks'] = []
    model_params['training_hooks'].append( hvd.BroadcastGlobalVariablesHook(0) )
    
    if 'optimizer' not in model_params: model_params['optimizer'] = tf.train.AdamOptimizer(model_params['learning_rate'])
    model_params['optimizer'] = hvd.DistributedOptimizer(model_params['optimizer'])
    
    if config == None:
        config = tf.estimator.RunConfig()
        
    config.session_config.gpu_options.visible_device_list = str(hvd.local_rank())
    
    return model_dir, model_params, config