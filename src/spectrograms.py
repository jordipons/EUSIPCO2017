import os
import librosa
from joblib import Parallel, delayed
import pickle
import glob
import json
import sys
import common

"""
spectrograms.py: computes spectrograms.

Requires pre-precomputing an 'index_file',a .tsv where an index with id,path is defined for a dataset.

The results and parameters of this script are stored in common.DATA_FOLDER/spectrograms/
'index.tsv' stores the 'id,path_spectrogram,path_audio'.
'path_spectrgram' and 'path_audio' absolute reference from common.DATA_FOLDER.

Step 1/5 of the pipeline.

"""
config = {
	'spectrograms_code_version': 'eusipco2017',
	'audio_folder' : 'audio/MagnaTagATune/', 	# end it with / !!!
	'spectrograms_name' : 'MagnaTT_dieleman_spectrograms',
	'original_sr' : 16000,
	'resample_sr' : 16000, # if one does not wish to resample, set resample_sr=original_sr
	'hop' : 256,
	'spectrogram_type' : 'mel',
	'n_fft' : 512,
	'n_mels' : 128,
	'convert_id' : False, 										# converts the (path) name of a file to its ID name - correspondence in index_file.
	'index_file' : 'index/index_MagnaTT.tsv',				# index to be converted. THIS IS THE LIST THAT ONE WILL COMPUTE
	'audio_ext' : ['mp3'] 										# in list form
}

# Choi et al.: 	'original_sr' : 16000, 'resample_sr' : 12000, 'hop' : 256, 'spectrogram_type' : 'mel', 'n_fft' : 512, 'n_mels' : 96.
# Dieleman et al.: 	'original_sr' : 16000, 'resample_sr' : 16000, 'hop' : 256, 'spectrogram_type' : 'mel', 'n_fft' : 512, 'n_mels' : 128.

num_process = 8
compute_spectro = True
path2id = dict()

def compute_spec(audio_file,spectro_file):
	# Get actual audio
	audio, sr = librosa.load(audio_file, sr=config['original_sr'])
	# resample?
	if config['original_sr']!=config['resample_sr']:
		audio = librosa.resample(audio, sr, config['resample_sr'])
		sr=config['resample_sr']
	# Compute spectrogram
	if config['spectrogram_type']=='cqt':
		spec = librosa.cqt(audio, sr=sr, hop_length=config['hop'], n_bins=config['cqt_bins'], real=False)
	elif config['spectrogram_type']=='mel':
		spec = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=config['hop'],n_fft=config['n_fft'],n_mels=config['n_mels'])
	elif config['spectrogram_type']=='stft':
		spec = librosa.stft(y=audio,n_fft=config['n_fft'])

	# Write results:
	with open(spectro_file, "w") as f:
		pickle.dump(spec, f, protocol=-1) # spec shape: MxN.

def do_process(id, audio_file, spectro_file):
	try:
		if compute_spectro:
			if not os.path.exists(spectro_file[:spectro_file.rfind('/')+1]):
				os.makedirs(spectro_file[:spectro_file.rfind('/')+1])
			compute_spec(audio_file,spectro_file)
			fw = open(common.DATA_FOLDER+config['spectro_folder']+"index.tsv","a")
			fw.write("%s\t%s\t%s\n" % (id,spectro_file[len(common.DATA_FOLDER):],audio_file[len(common.DATA_FOLDER):]))
			fw.close()
			print 'Computed spec: %s' % audio_file
		else:
			if os.path.isfile(spectro_file):
				fw = open(config['spectro_folder']+"index.tsv","a")
				fw.write("%s\t%s\t%s\n" % (id,spectro_file[len(common.DATA_FOLDER+config['spectro_folder']):],audio_file[len(common.DATA_FOLDER+config['audio_folder']):]))
				fw.close()
	except Exception as e:
		ferrors = open(common.DATA_FOLDER+config['spectro_folder']+"errors.txt","a")
		ferrors.write(audio_file+"\n")
		ferrors.write(str(e))
		ferrors.close()
		print 'Error computing spec', audio_file
		print str(e)

def process_files(files):
	Parallel(n_jobs=num_process)(delayed(do_process)(id, audio_file, spectro_file)
						   for id, audio_file, spectro_file in files)
	## Debug ##
	#print 'WARNING: Parallelization is not used!'
	#for id, audio_file, spectro_file in files:
	#	do_process(id, audio_file, spectro_file)

if __name__ == '__main__':

	# set spectrograms folder
	config['spectro_folder'] = "spectrograms/spectro_%s_%s_%s/" % (config['spectrograms_name'],config['spectrogram_type'],config['spectrograms_code_version'])
	if not os.path.exists(common.DATA_FOLDER+config['spectro_folder']):
		os.makedirs(common.DATA_FOLDER+config['spectro_folder'])
	else:
		sys.exit("EXIT: already exists a folder with this name!\nIf you need to compute those again, remove folder.")

	# create empty spectrograms index
	fw = open(common.DATA_FOLDER+config['spectro_folder']+"index.tsv","w")
	fw.close()

	# list audios to process: according to 'index_file'
	files_to_convert = []
	f=open(common.DATA_FOLDER+config["index_file"])
	for line in f.readlines():
		id, audio = line.strip().split("\t")
		if config['convert_id']:
			spect = id+".pk"
		else:
			spect = audio[:audio.rfind(".")]+".pk"
		files_to_convert.append((id,common.DATA_FOLDER+config['audio_folder']+audio,common.DATA_FOLDER+config['spectro_folder']+spect))

	print str(len(files_to_convert))+' audio files to process!'

	# compute spectrogram
	process_files(files_to_convert)

	# save parameters
	json.dump(config, open(common.DATA_FOLDER+config['spectro_folder']+"params.json","w"))

	print "Spectrograms folder: "+common.DATA_FOLDER+config['spectro_folder']

# COMMENTS:

## pickle protocol=-1?

## convert_id == FALSE: creates sub-directories - put to false for magna.
## convert_id == TRUE: does not creates sub-directories - in some cases one does not care.