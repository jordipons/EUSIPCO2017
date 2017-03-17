from random import shuffle
import json
import common
import os
import csv

"""
exp_setup.py: splits the spectrograms data in train, val, test.
Different experiments can be set up using the same spectrograms,
but with different ground truth.

Requires the having computed spectrograms (with 'spectrograms.py') in 'spectro_folder'.
Requires pre-precomputing a 'gt_all',a .tsv where an index with id,gt is defined for a dataset.

The results and parameters of this script are stored in common.DATA_FOLDER/exp_setup/
It generates the splits with 'id, spectrogram_path, ground_truth'.

Step 2/5 of the pipeline.

NOT deterministic experiment: if random shuffle.

"""
config = {
	'exp_setup_code_version': 'eusipco2017',
	'setup_name' : 'dieleman_setup',
	'spectro_folder' : 'spectrograms/spectro_MagnaTT_dieleman_spectrograms_mel_eusipco2017/', # end it with / !!!
	'gt_all' : 'index/gt_MagnaTT.tsv'
}

def split_dataset(ids):
	if config['shuffle']:
		shuffle(ids)
	n_train = int(len(ids)*config['trainset_size'])
	n_val = int(len(ids)*config['valset_size'])
	return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]

def split_magna(ids):
	train_set = []
	val_set = []
	test_set = []
	for id in ids:
		path = id2audiopath[id]
		folder = int(path[path.rfind("/")-1:path.rfind("/")],16)
		if folder < 12:
			train_set.append(id) # 0,1,2,3,4,5,6,7,8,9,a,b
		elif folder < 13:
			val_set.append(id) # c
		else:
			test_set.append(id) # d,e,f
	return train_set,val_set,test_set

def save_set(filename,itemset,id2specpath,id2gt):
	fw = open(filename,"w")
	for item in itemset:
		fw.write("%s\t%s\t%s\n" % (item, id2specpath[item], id2gt[item]))
	fw.close()

def load_id2gt(gt_file):
	ids = []
	fgt = open(gt_file)
	id2gt = dict()
	for line in fgt.readlines():
		id, gt = line.strip().split("\t")
		id2gt[id] = gt
		ids.append(id)
	return ids,id2gt

def load_id2audiopath(index_file):
	f=open(index_file)
	audiopath2id = dict()
	for line in f.readlines():
		id,path = line.strip().split("\t")
		audiopath2id[id] = path
	return audiopath2id

def load_id2specpath(index_file):
    fspec = open(index_file)
    id2specpath = dict()
    for line in fspec.readlines():
        id, path, _ = line.strip().split("\t")
        id2specpath[id] = path
    return id2specpath

if __name__ == "__main__":

	# load parameters from previous processing step: 'spectrograms.py'
	params = json.load(open(common.DATA_FOLDER+config['spectro_folder']+"params.json"))
	config['spectro_params'] = params

	# load index to spectrogram/audio path
	id2specpath=load_id2specpath(common.DATA_FOLDER+config['spectro_folder']+"index.tsv")
	id2audiopath=load_id2audiopath(common.DATA_FOLDER+config['spectro_params']['index_file'])

	# as a result of this step: id2gt, id2path, and lists of id's for each partition.
	if config['gt_all'] != '':
		_,id2gt=load_id2gt(common.DATA_FOLDER+config['gt_all'])

		ids = set(id2specpath.keys()).intersection(set(id2gt.keys())) # still 'index_file' rules, 'index_gt' with all annotations.
		if 'magna' in config['spectro_params']['spectrograms_name'].lower():
			trainset, valset, testset = split_magna(ids)
		else:
			trainset, valset, testset = split_dataset(ids)
	else:
		trainset,id2gt_train=load_id2gt(common.DATA_FOLDER+config['gt_train'],id2specpath)
		print len(id2gt_train)
		valset,id2gt_val=load_id2gt(common.DATA_FOLDER+config['gt_val'],id2specpath)
		print len(id2gt_val)
		testset,id2gt_test=load_id2gt(common.DATA_FOLDER+config['gt_test'],id2specpath)
		print len(id2gt_test)
		id2gt = id2gt_train.copy()
		id2gt.update(id2gt_val)
		id2gt.update(id2gt_test)
		ids = set(id2specpath.keys()).intersection(set(id2gt.keys())) #  do we really need that?

	# save experimental setup
	config['numOutputNeurons']=len(eval(id2gt[id2gt.keys()[0]]))
	print config
	config['setup_folder'] = "exp_setup/%s_%s/" % (config["setup_name"],config['exp_setup_code_version'])
	if not os.path.exists(common.DATA_FOLDER+config['setup_folder']):
		os.makedirs(common.DATA_FOLDER+config['setup_folder'])
	save_set(common.DATA_FOLDER+config['setup_folder']+"trainset.tsv",trainset,id2specpath,id2gt)
	save_set(common.DATA_FOLDER+config['setup_folder']+"valset.tsv",valset,id2specpath,id2gt)
	save_set(common.DATA_FOLDER+config['setup_folder']+"testset.tsv",testset,id2specpath,id2gt)
	json.dump(config, open(common.DATA_FOLDER+config['setup_folder']+"params.json","w"))
	print 'Experimental setup folder: '+common.DATA_FOLDER+str(config['setup_folder'])
