import pickle
import glob
import numpy as np
import os
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import common
import time

"""
patches.py: computes the patches and normalizes de data.

Requires the previous run of 'exp_setup.py'.

The results and parameters of this script are stored in common.DATA_FOLDER/patches/

Step 3/5 of the pipeline.

NOT deterministic experiment: random sampling and randomize training examples.

"""
config = {
	'patches_code_version': 'elementWise_memory',
	'setup' : 'dieleman_setup_eusipco2017', 							#experimental setup name in 'exp_setup.py'
	'n_frames': '', 									# if '', compute n_frames from 'window', SET AS INT otherwise!
	'window' : 3, 										# in seconds
	'spec_processing' : 'logC', #'squared_log' or 'logC'
	'normalization' : 'element_wise',		#'global' or 'element_wise'
	'mean_var_type' : 'memory',   # 'calculus' or 'memory'
	'mode_sampling' : 'overlap_sampling',
	'param_sampling' : 187 				#if 'overlap_sampling': param_sampling=hop_size
	}														#if 'random': param_sampling=number of samples

def sample_spec(spec,n_frames,mode_sampling,param_sampling):
	'''
	spec: input spectrogram to sample from
	n_frames: length of the sample
	mode_sampling: 'overlap_sampling' or 'random'
	param_sampling: 	if 'overlap_sampling': param_sampling=hop_size
						if 'random': param_sampling=number of samples
	'''
	if mode_sampling=='overlap_sampling':
		for i in xrange(0,(int((spec.shape[0]-int(n_frames))/param_sampling)*param_sampling)+1,param_sampling):
			yield spec[i:i+int(n_frames)]
	else: # random sampling
		for i in range(0,param_sampling):
			r_idx = np.random.randint(0, high=int(spec.shape[0]) - int(n_frames) + 1)
			yield spec[r_idx:r_idx + int(n_frames)]

def create_patches(set_file,set_type):
    # Create patches folder for 'set_type': train, test, val.
	patches_folder = common.DATA_FOLDER+config["patches_folder"] + set_type + "/"
	if not os.path.exists(patches_folder):
		os.makedirs(patches_folder)
	# Get exp_setup partition: train, test, val.
	df_items = pd.read_table(common.DATA_FOLDER+set_file, usecols=[0, 1, 2], names=['id', 'path', 'gt'])
	# Create patches from spectrograms
	n = 0
	fw = open(patches_folder + "all_patches_ids.tsv","w") # create indexing file where all patches are saved
	for index, row in df_items.iterrows():
		# load spectrogram
		spec = pickle.load(open(common.DATA_FOLDER+row['path']))
		# normalize amplitude
		# transpose to x,y = NxM instead of MxN.
		if config['spec_processing'] == 'squared_log':
			spec = common.logamplitude(np.abs(spec) ** 2).T#,ref_power=np.max)!!!!!!!!
		elif config['spec_processing'] == 'logC':
			spec = np.log10(10000*spec+1).T
		# save
		if int(spec.shape[0]) >= int(config['n_frames']):
			sample_counter=0
			for sample in sample_spec(spec,int(config['n_frames']),config['mode_sampling'],config['param_sampling']):
				try:
					patch_path=config["patches_folder"] + set_type + "/"+row['path'].split("/")[-2]+"/"
					if not os.path.exists(common.DATA_FOLDER+patch_path):
						os.makedirs(common.DATA_FOLDER+patch_path)
					patch_path=patch_path+row['path'][row['path'].rfind("/")+1:row['path'].rfind(".")]+'_'+str(sample_counter)+'.npy'
					fw.write("%s\t%s\t%s\t%s\n" % (row['id'],row['path'],sample_counter,patch_path)) # id, spectro_path, sample_counter, patch_path
					# patches have NxM dimensions.
					np.save(common.DATA_FOLDER+patch_path, np.asarray(sample)) #!!!!!!! asarray????
					sample_counter=sample_counter+1
				except Exception,e:
					print str(e)
		n+=1 # n is for tracking progress
		if n%100==0:
			print n

	return patches_folder

def get_scaler(folder):
	if config['mean_var_type']=='memory':
		return get_scaler_memory(folder)
	elif config['mean_var_type']=='calculus' and config['normalization']=='global':
		return get_scaler_calculus(folder)
	else:
		print "ERROR: set 'normalization' or 'mean_var_type' correctly."

def get_scaler_calculus(folder):
	total_mu=0
	total_var=0
	total_l=0
	# Load patches
	df_items = pd.read_table(folder+'all_patches_ids.tsv', usecols=[0, 1, 2, 3], names=['id', 'spectro_path', 'sample_count', 'patch_path'])
	for index, row in df_items.iterrows():
		if index%100==0:
			print str(index+1)+'/'+str(df_items.shape[0])
		patch_file=common.DATA_FOLDER+row['patch_path']
		patch = np.load(patch_file)
		# Normalize patches according to scaler
		patch=patch.reshape(patch.shape[0] * patch.shape[1])
		mu=np.mean(patch)
		v=np.var(patch)
		l=len(patch)
		total_mu=(l*mu+total_l*total_mu)/(l+total_l)
		a=l*((mu*mu)+v)
		b=total_l*((total_mu*total_mu)+total_var)
		c=(a+b)/(l+total_l)
		total_var=c-(total_mu*total_mu)
		total_l=l+total_l

	return [total_mu,total_var]

def get_scaler_memory(folder):
	# Load all patches for computing a BIASED estimation of the mean and var
	df_items = pd.read_table(folder+'all_patches_ids.tsv', usecols=[0, 1, 2, 3], names=['id', 'spectro_path', 'sample_count', 'patch_path'])
	counter=0
	for index, row in df_items.iterrows():
		if index%100==0:
			print str(index+1)+'/'+str(df_items.shape[0])
		patch_file=common.DATA_FOLDER+row['patch_path']

		if counter==0:
			patches=np.load(patch_file)
			counter=counter+1
		else:
			patches=np.concatenate((patches,np.load(patch_file)),axis=0)
			counter=counter+1

		if counter==2000:
			break

	if config['normalization']=='global':
		patches=patches.reshape(patches.shape[0] * patches.shape[1])

	scaler = StandardScaler()
	scaler.fit(patches)
	## Check numerically mean/std normalization ##
	t_patches=scaler.transform(patches)
	print '\t- Mean @ visible patches: '+str(t_patches.mean())
	print '\t- Std @ visible patches: '+str(t_patches.std())
	print '\t- Var @ visible patches: '+str(t_patches.var())

	return scaler

def save_normalized_patches(folder, scaler):
	df_items = pd.read_table(folder+'all_patches_ids.tsv', usecols=[0, 1, 2, 3], names=['id', 'spectro_path', 'sample_count', 'patch_path'])
	for index, row in df_items.iterrows():
		patch_file=common.DATA_FOLDER+row['patch_path']
		# Load patch
		patch = np.load(patch_file)
		shape = patch.shape
		if config['normalization']=='global':
			patch=patch.reshape(patch.shape[0] * patch.shape[1])
		# Normalize patches according to scaler
		if config['mean_var_type']=='memory':
			patch=scaler.transform(patch)
		elif config['mean_var_type']=='calculus':
			patch = (patch-scaler[0])/scaler[1]
		patch = patch.reshape(1,1,shape[0],shape[1])
		np.save(patch_file, patch) # save patch with dimensions (1,1,N,M)

if __name__ == '__main__':

	# Load parameters from previous processing steps: 'spectrograms.py' and 'exp_setup.py'.
	params = json.load(open(common.DATA_FOLDER+"exp_setup/%s/params.json" % config["setup"]))
	config['spectro_params'] = params['spectro_params']
	config['setup_params'] = params
	config['setup_params'].pop('spectro_params',None) 	# remove repeated variables

	# Set patch parameters
	if config['n_frames'] == '':
		config['n_frames'] = int(config['window'] * config['spectro_params']['resample_sr'] / float(config['spectro_params']['hop']))
		print 'Number of frames: '+str(config['n_frames'])
	config['patches_folder'] = 'patches/patches_%s_%s_%s_%s_%s/' % (config['setup'],config['n_frames'],config['spec_processing'],config['patches_code_version'],int(time.time()))
	config['xInput']=config['n_frames']
	if config['spectro_params']['spectrogram_type']=='cqt':
		config['yInput']=config['spectro_params']['cqt_bins']
	elif config['spectro_params']['spectrogram_type']=='mel':
		config['yInput']=config['spectro_params']['n_mels']

	print config

	print '- Generating training set..'
	patches_folder_train = create_patches(config['setup_params']['setup_folder']+"trainset.tsv","train")
	scaler = get_scaler(patches_folder_train) # compute scaler in training data
	#if config['mean_var_type']=='memory':
	#	print '\tScalar-mean: '+str(scaler.mean_)
	#	print '\tScalar-var: '+str(scaler.var_)
	#elif config['mean_var_type']=='calculus':
	#	print '\tScalar-mean: '+str(scaler[0])
	#	print '\tScalar-var: '+str(scaler[1])
	save_normalized_patches(patches_folder_train, scaler)

	print '- Generating validation set..'
	patches_folder_val = create_patches(config['setup_params']['setup_folder']+"valset.tsv","val")
	save_normalized_patches(patches_folder_val, scaler)

	print '- Generating test set..'
	patches_folder_test = create_patches(config['setup_params']['setup_folder']+"testset.tsv","test")
	save_normalized_patches(patches_folder_test, scaler)

	# Save scaler and parameters
	json.dump(config, open(common.DATA_FOLDER+config["patches_folder"]+"params.json","w"))
	pickle.dump(scaler,open(common.DATA_FOLDER+config["patches_folder"]+"scaler.pk",'wb'))

	print 'Patches folder: '+str(config['patches_folder'])

# DOUBTS
## std? var? how to compute?

## ref_power=np.max!

## Transpose in:
# 		if config['spec_processing'] == 'squared_log':
#			spec = common.logamplitude(np.abs(spec) ** 2).T#,ref_power=np.max)!!!!!!!!
#		elif config['spec_processing'] == 'logC':
#			spec = np.log10(10000*spec+1).T
