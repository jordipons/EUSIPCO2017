import time, random
import numpy as np
from sklearn import metrics

import pandas as pd
import common
import os
import json
import csv

import theano, lasagne
import theano.tensor as T

import build_architecture as buildArch

"""
train.py: trains a deep learning model defined in 'builid_architecture.py'.

Requires the previous run of 'patches.py'.

The results and parameters of this script are stored in common.DATA_FOLDER/train/

Step 4/5 of the pipeline.

NOT deterministic experiment: if random initialization.

"""
config = {
	'train_code_version': 'v0',
    'patches' : 'patches/patches_dieleman_setup_eusipco2017_187_logC_elementWise_memory_1489997707/', # end it with / !!!
    'random_seed' : 0, # 0 to disable.
    'saveAllModels' : False,

    'type' : 'proposed2', #'smallSquared' or 'proposed' or 'proposed2'
    'num_epochs' : 200,
    'batchSize' : 10,
    'optimization' : 'schedule', #'schedule' or 'adam'
    'initialize_model' : 'random', # 'random' or model path.
    'lr' : 0.1, # if 'schedule', otherwise useless.
    'momentum' : 0, # if 'schedule', otherwise useless.
}

def iterate_minibatches(config,idx_id_patches_in, batchsize, mode):

    if mode=='train':
        keys = idx_id_patches_in.keys()
        random.shuffle(keys)
        idx_id_patches=dict()
        i=0
        for key in keys:
            idx_id_patches[i]= idx_id_patches_in[key]
            i=i+1
    else:
        idx_id_patches=idx_id_patches_in

    id2gt=dict()
    df_items = pd.read_table(common.DATA_FOLDER+config['setup_params']['setup_folder']+mode+'set.tsv', usecols=[0, 1, 2], names=['id', 'path', 'gt'])
    for index, row in df_items.iterrows():
        id=row['id']
        gt = eval(row['gt'])
        id2gt[id] = gt

    for start_idx in range(0, len(idx_id_patches) - batchsize + 1, batchsize):          # BE CAREFUL! if the last examples are not enough to create a batch, are descarted.
        D = np.zeros(batchsize*int(config['patches_params']['yInput'])*int(config['patches_params']['xInput']),dtype=np.float32).reshape(batchsize,1,int(config['patches_params']['xInput']),int(config['patches_params']['yInput']))
        A = np.zeros((batchsize,config['setup_params']['numOutputNeurons']),dtype=np.uint8)+9
        ID = np.zeros(batchsize)+common.ERROR_CODE
        PN = np.zeros(batchsize)+common.ERROR_CODE
        i=0
        for idx in range(start_idx,start_idx + batchsize,1):
            id=idx_id_patches[idx][0]
            patch=np.load(common.DATA_FOLDER+idx_id_patches[idx][3])
            D[i]=patch.reshape(1,1,int(config['patches_params']['xInput']),int(config['patches_params']['yInput']))
            A[i]=map(np.uint8,id2gt[id])
            ID[i]=id
            PN[i]=idx_id_patches[idx][2]
            i+=1
        yield D, A, ID, PN

def index_patches(all_patches_id_path):
    all_patches_id = pd.read_table(all_patches_id_path, usecols=[0,1,2,3], names=['id', 'spec_path','sample','patch_path'])
    idx_id_patches = dict()
    for index, row in all_patches_id.iterrows():
        idx_id_patches[index]=[row['id'], row['spec_path'], row['sample'], row['patch_path']]
    return idx_id_patches

def compileFn(network,input_var,target_var):
    weight_decay = 1e-5 # http://luizgh.github.io/libraries/2015/12/08/getting-started-with-lasagne/
    # define training functions
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    train_loss = loss.mean() + weight_decay*weightsl2
    params = lasagne.layers.get_all_params(network, trainable=True)
    if config['optimization'] == 'adam':
        updates = lasagne.updates.adam(train_loss, params)
    else:
        updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=config['lr'], momentum=config['momentum'])

    # define testing/val functions
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    tst_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
    test_loss = tst_loss.mean()

    # compile training and test/val functions
    train_fn = theano.function([input_var, target_var], [train_loss], updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss])
    predict_fn = theano.function([input_var], test_prediction)

    return train_fn,val_fn,predict_fn

def buildNet(input_var,target_var,config):
    network,netLayers = buildArch.buildNet(input_var,config)

    config['numParamArchitecture'] = lasagne.layers.count_params(network)
    print '\t Number of weights: ' + str(config['numParamArchitecture'])

    train_fn,val_fn,predict_fn = compileFn(network,input_var,target_var)

    return train_fn,val_fn,predict_fn,network,config

if __name__ == '__main__':

    print(" - Setting the environment..")

    if config['random_seed']!=0:
        np.random.seed(config['random_seed'])

    results_folder=common.DATA_FOLDER+'train/'
    if not os.path.exists(results_folder):
	    os.makedirs(results_folder)

	# load parameters from previous processing step: 'spectrograms.py', 'exp_setup.py' and 'patches.py'
    params = json.load(open(common.DATA_FOLDER+config['patches']+'params.json'))
    config['spectro_params'] = params['spectro_params']
    config['setup_params'] = params['setup_params']
    config['patches_params'] = params
    config['patches_params'].pop('spectro_params',None) # remove repeated variables
    config['patches_params'].pop('setup_params',None) # remove repeated variables

    print '\t'+str(config)

    print(" - Fetching data..")

    idx_id_patches_train=index_patches(common.DATA_FOLDER+config['patches']+'train/all_patches_ids.tsv')
    num_train_patches=len(idx_id_patches_train)
    print '\tNumber of training examples (patches): '+str(num_train_patches)
    idx_id_patches_val=index_patches(common.DATA_FOLDER+config['patches']+'val/all_patches_ids.tsv')
    num_val_patches=len(idx_id_patches_val)
    print '\tNumber of validation examples (patches): '+str(num_val_patches)
    # in case we remove AUC computation, this can be placed inside iterate_minibatch.

    print(" - Building network..")

    input_var = T.tensor4('inputs')
    target_var = T.imatrix('targets')
    train_fn,val_fn,predict_fn,network,config=buildNet(input_var,target_var,config)

    # load model for initialization
    if config['initialize_model'] != 'random':
        with np.load(config['initialize_model']) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)

    print(" - Training..")

    hash = random.getrandbits(128)
    name_model=config['patches_params']['setup']+'_'+config['type']+'_'+config['train_code_version']+'_'+str(hash)

    valueAns_earlyStop=np.inf
    countRaising=0
    valueAns_pickBest=np.inf
    for epoch in range(config['num_epochs']):
        # training set
        start_time = time.time()
        train_err = []
        for batch in iterate_minibatches(config,idx_id_patches_train, config['batchSize'],'train'):
            inputs, targets, ids, patch_number = batch
            # compute error
            err = train_fn(inputs, targets)
            train_err.append(err[0])

        # validation set: individual strategy, equal as in training
        val_err = []
        for batch in iterate_minibatches(config,idx_id_patches_val, config['batchSize'],'val'):
            inputs, targets, ids, patch_number = batch
            # compute error
            err = val_fn(inputs, targets)
            val_err.append(err[0])

        # output
        print("    Epoch {} of {} took {:.3f}s".format(
            epoch + 1, config['num_epochs'], time.time() - start_time))
        print("      training loss:\t\t\t\t{:.6f}".format(np.mean(train_err)))
        print("      validation loss:\t\t\t\t{:.6f}".format(np.mean(val_err)))

        if config['optimization'] == 'schedule':   # learning rate schedule, with early stopping. It requires long runs, ie: #epochs = 2000.

            if np.mean(train_err) > valueAns_earlyStop:
                countRaising=countRaising+1
                print 'Counter raised: '+str(countRaising)
                if countRaising>10:
                    break
            else:
                if countRaising>3:
                    config['lr']=config['lr']/2
                    print 'Compiling..'
                    train_fn,val_fn,predict_fn=compileFn(network,input_var,target_var)
                    print 'Learning rate changed: '+str(config['lr'])
                    countRaising=0
                valueAns_earlyStop=np.mean(train_err)

        elif config['optimization'] == 'adam': # early stopping

            if np.mean(train_err) > valueAns_earlyStop:
                countRaising=countRaising+1
                print '\t  # Counter raised: '+str(countRaising)
                if countRaising>20:
                    break
            else:
                valueAns_earlyStop=np.mean(train_err)

        # storing data: tracking the training
        if config['saveAllModels']:
            np.savez(results_folder+name_model+'_iteration_'+str(epoch + 1), *lasagne.layers.get_all_param_values(network))
        # save the best model
        if (np.mean(val_err))<valueAns_pickBest:
            valueAns_pickBest=np.mean(val_err)
            np.savez(results_folder+name_model, *lasagne.layers.get_all_param_values(network))
            res = open(results_folder+name_model+'.result', 'w')
            res.write("    Epoch {} of {} took {:.3f}s\n".format(epoch + 1, config['num_epochs'], time.time() - start_time))
            res.write("      training loss:\t\t\t\t{:.6f}\n".format(np.mean(train_err)))
            res.write("      validation loss:\t\t\t\t{:.6f}\n".format(np.mean(val_err)))
            res.close()
        # save config
        if epoch==0:
            json.dump(config, open(results_folder+name_model+'.param','w'))
            tr = open(results_folder+name_model+'.training', 'w')
            tr.write('epoch,train_err,val_err\n')
            tr.close()
        # save training evolution
        tr = open(results_folder+name_model+'.training', 'a')
        tr.write(str(epoch)+','+str(np.mean(train_err))+','+str(np.mean(val_err))+'\n')
        tr.close()

    print 'Model name: '+str(results_folder+name_model)
