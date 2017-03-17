import numpy as np
import common, train
import theano.tensor as T
import lasagne
from sklearn import metrics
import os, time, json

"""
test.py: evaluates how the trained model performs.

Requires the previous run of 'train.py'.

The results and parameters of this script are stored in common.DATA_FOLDER/test/

Step 5/5 of the pipeline.

"""
test_params = {
    'partition' : 'test', # para ver en train cuando hace!! para medir overfitting.
    'model_name' : ['dieleman_setup_eusipco2017_proposed2_v0_128172504761355761965820505631639987477'],
    'test_name' : 'TEST_'+str(int(time.time()))
}

def compute_auc(estimated,true):
    '''
    AUC is computed at the tag level because there are many songs in the MTT having no annotations - zeros array.
    Input dimensions:
    - estimated: #songs x #outputNeurons
    - true: #songs x #annotations
    where #outputNeurons = #annotations
    '''
    aucs=[]
    for count in range(estimated.shape[1]-1):
        if np.min(true[:,count]) != np.max(true[:,count]):
            auc = metrics.roc_auc_score(true[:,count],estimated[:,count])
            aucs.append(auc)
        else:
            print 'WARNING: All 0s or 1s, can not compute AUC! Tag #'+str(count)
    return np.mean(aucs)

if __name__ == '__main__':

    print 'Number of models to test: '+str(len(test_params['model_name']))

    first=True
    for model in test_params['model_name']:

        print '\nMODEL: '+model

        print(" - Set environment..")
        # load parameters from previous processing step: 'spectrograms.py', 'exp_setup.py' and 'patches.py'
        config = json.load(open(common.DATA_FOLDER+'train/'+model+'.param'))
        config['test_params'] = test_params

        print '\t'+str(config)

        idx_id_block_test=train.index_patches(common.DATA_FOLDER+config['patches']+test_params['partition']+'/all_patches_ids.tsv')
        num_test_patches=len(idx_id_block_test)
        print 'Number of testing examples (patches): '+str(num_test_patches)

        if first:
            patch_labels=np.empty((0,2+config['setup_params']['numOutputNeurons']*2))
            # id [0], patch-number [1], prediction [2:numOutputNeurons+2], target [numOutputNeurons+2:]
            first=False

        print(" - Building network..")

        input_var = T.tensor4('inputs')
        target_var = T.imatrix('targets')
        train_fn,val_fn,predict_fn,network,config=train.buildNet(input_var,target_var,config)

        print(" - Loading model..")

        # load model
        with np.load(common.DATA_FOLDER+'train/'+model+'.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)

        print(" - Running predictions..")

        # test set: predict patches
        test_err = []
        test_auc = []
        batch_counter=0
        for batch in train.iterate_minibatches(config,idx_id_block_test, config['batchSize'],'test'):
            inputs, targets, ids, patch_number = batch
            batch_counter=batch_counter+config['batchSize']
            if batch_counter%1000==0:
                print str(batch_counter)+'/'+str(num_test_patches)
            # compute error
            err = val_fn(inputs, targets)
            test_err.append(err[0])
            # compute prediction
            prediction = predict_fn(inputs)
            for c in range(0,len(targets),1): # DO THIS BY MATRICES intead of using a for.
                tmp=np.array([ids[c],patch_number[c]])
                tmp=np.append(tmp,targets[c])
                tmp=np.append(tmp,prediction[c])
                patch_labels=np.append(patch_labels,tmp.reshape(1,2+config['setup_params']['numOutputNeurons']*2),axis=0)

    print '\nRUN EVALUATION:'
    # average with patches having same id
    patch_labels_avg=np.empty((0,1+config['setup_params']['numOutputNeurons']*2))# id, prediction, target
    ids_list=np.unique(patch_labels[:,0])
    for id in ids_list:
        idx=np.where(patch_labels[:,0]==id)
        tmp=np.array([id])
        tmp=np.append(tmp,patch_labels[idx[0][0],2:2+config['setup_params']['numOutputNeurons']])
        tmp=np.append(tmp,np.average(patch_labels[idx,2+config['setup_params']['numOutputNeurons']:][0],axis=0))
        patch_labels_avg=np.append(patch_labels_avg,tmp.reshape(1,1+config['setup_params']['numOutputNeurons']*2),axis=0)

    # compute individual auc
    auc_individual = compute_auc(patch_labels[:,2+config['setup_params']['numOutputNeurons']:],patch_labels[:,2:2+config['setup_params']['numOutputNeurons']])
    auc_avg = compute_auc(patch_labels_avg[:,1+config['setup_params']['numOutputNeurons']:],patch_labels_avg[:,1:1+config['setup_params']['numOutputNeurons']])

    # output
    print("   Final results:")
    print("      test loss:\t\t\t\t\t{:.6f}".format(np.mean(test_err)))
    print("      patch level - test auc:\t\t{:.4f}".format(auc_individual))
    print("      [AVG] song level - test auc:\t{:.4f}".format(auc_avg))
    print("      number of weights: \t\t\t{:.0f}".format(config['numParamArchitecture']))

    # storing data: tracking the results
    test_folder = common.DATA_FOLDER+'test/'
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    res = open(test_folder+config['test_params']['test_name']+'_'+config['test_params']['partition']+'.result', 'a')

    res.write("\nFinal results:\n")
    res.write("      test loss:\t\t\t\t\t{:.6f}\n".format(np.mean(test_err)))
    res.write("      patch level - test auc:\t\t{:.4f}\n".format(auc_individual))
    res.write("      [AVG] song level - test auc:\t{:.4f}\n".format(auc_avg))
    res.write("      number of weights: \t\t\t{:.0f}\n".format(config['numParamArchitecture']))

    json.dump(test_params, open(test_folder+config['test_params']['test_name']+'_'+config['test_params']['partition']+'.param','w'))
