import random
import os
import sys
import argparse
import copy
import shutil
import json
from pprint import pprint
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import uproot3 as uproot


# choich
def F1(arr, target):
    tp=0
    fp=0
    fn=0
    for label in range(3):
        for pred in range(3):
            if label == pred == target:
                tp += arr[pred][label]
            elif label != target and pred == target:
                fp += arr[pred][label]
            elif label == target and pred != target:
                fn += arr[pred][label]
    return 2 * tp / (2 * tp + fp + fn)


"""
How To:
Example for running from command line:
python <path_to>/SetToGraph/main_scripts/main_jets.py
"""
# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from models.set_to_graph import SetToGraph
from models.set_to_graph_siam import SetToGraphSiam
from dataloaders import jets_loader
#from performance_eval.eval_test_jets import eval_jets_on_test_set
from models.classifier import JetClassifier


DEVICE = 'cpu'#'cuda' #NOT USING CUDA


def parse_args():
	"""
	Define and retrieve command line arguements
	:return: argparser instance
	"""
	argparser = argparse.ArgumentParser(description=__doc__)
	
	argparser.add_argument('-b', '--bs', default=1000, type=int, help='Batch size to use')
												#2048

	argparser.add_argument('--vertexing_model_type')
	argparser.add_argument('--path_to_trained_model', default=None, help='path to trained model')
	argparser.add_argument('--outputfilename')

	argparser.add_argument('--use_rave', dest='use_rave', action='store_true')
	argparser.set_defaults(use_rave=False)

	# b-jet tagging weights
	argparser.add_argument('--alpha', default=1.0, type=float, help='Weight for b/c (b > alpha * c)')
	argparser.add_argument('--beta', default=1.0, type=float, help='Weight for b/l (b > beta * l)')

	args = argparser.parse_args()



	return args









def evaluate(data, model,use_rave=False):

	model.eval()


	all_jet_predictions = []

	for batch in data:

		if use_rave:
			sets, _, _, jet_features, jet_label, rave_input = batch
			sets = sets.to(DEVICE, torch.float)
			jet_prediction = model(jet_features,sets,rave_input).cpu().data.numpy() # B,N,N
		else:
			sets, _, _, jet_features, jet_label = batch
			sets = sets.to(DEVICE, torch.float)
			jet_prediction = model(jet_features,sets).cpu().data.numpy() # B,N,N
		
		
		
	  
		
		all_jet_predictions.append(jet_prediction)
   

	return np.concatenate(all_jet_predictions)


def main():
	start_time = datetime.now()

   

	config = parse_args()

	# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
	os.environ["CUDA_VISIBLE_DEVICES"] = '1' #config.gpu
	#torch.cuda.set_device(int(config.gpu))



	# Load data
	print('Loading test data...', end='', flush=True)
	test_data = jets_loader.JetGraphDataset('test',debug_load=False,add_jet_flav=True,add_rave_file=config.use_rave)
   
											#validation
	# Create model instance
	if config.vertexing_model_type == 'rnn':
		vertexing_config = {
			'in_features' : 10,
			'out_features' :1,
			'set_fn_feats' : [256, 256,128, 6],
			'method' : 'lin5',
			'hidden_mlp' : [256],
			'predict_diagonal' : False,
			'attention' : True,
			'set_model_type' : 'RNN'
			
		}
	elif config.vertexing_model_type == 'siam':
		vertexing_config = {
			'in_features' : 10,
			'set_fn_feats' : [384, 384, 384, 384, 5],
			'hidden_mlp' : [256],
		}
		
	elif config.vertexing_model_type == 's2g':
		vertexing_config = {
			'in_features' : 10,
			'out_features' :1,
			'set_fn_feats' : [256, 256, 256, 256, 5],
			'method' : 'lin5',
			'hidden_mlp' : [256],
			'predict_diagonal' : False,
			'attention' : True,
			'set_model_type' : 'deepset'
			
		}
	else:
		vertexing_config = {}
	
	model = JetClassifier(10,vertexing_config,vertexing_type=config.vertexing_model_type)
	model.load_state_dict( torch.load(config.path_to_trained_model,map_location='cpu') )
	model.eval()




	
	model = model.to(DEVICE)
   
	predictions = []
	indx_list = []

	max_batch_size = 1000

	for tracks_in_jet in tqdm(range(2, np.amax(test_data.n_nodes)+1)):
		trk_indxs = np.where(np.array(test_data.n_nodes) == tracks_in_jet)[0]
		if len(trk_indxs) < 1:
			continue
		indx_list += list(trk_indxs)


		n_sub_batches = len(trk_indxs)//max_batch_size+1

		sub_batches = np.array_split(trk_indxs,n_sub_batches)
		for sub_batch in sub_batches:
			if config.use_rave:
				sets = []
				jet_features = []
				rave_inputs = []

				for i in sub_batch:
					sets_i, _, _, jet_features_i, _, rave_input = test_data[i]
					sets.append(torch.tensor(sets_i))
					jet_features.append(torch.tensor(jet_features_i))
					rave_inputs.append(rave_input)

				sets = torch.stack(sets)  # shape (B, N_i, 10)
				jet_features = torch.stack(jet_features)
				rave_inputs = torch.stack(rave_inputs)
				with torch.no_grad():
					jet_predictions = model(jet_features,sets,rave_inputs)

			else:
				sets = []
				jet_features = []
				for i in sub_batch:
					sets_i, _, _, jet_features_i, _ = test_data[i]
					sets.append(torch.tensor(sets_i))
					jet_features.append(torch.tensor(jet_features_i))
					
				sets = torch.stack(sets)  # shape (B, N_i, 10)
				jet_features = torch.stack(jet_features)
				with torch.no_grad():
					jet_predictions = model(jet_features,sets)
			predictions += list(jet_predictions.cpu().data.numpy())

	# predictions <-> flavor compare

	sorted_predictions = [x for _, x in sorted(zip(indx_list, predictions))]

	# softmax
	# alpha = 7.02637 # if ( b > alpha * c &&
	# beta = 0.0157093 # b > beta * l ) then b-jet
	softmax_predictions = F.softmax(torch.tensor(sorted_predictions), dim=1)
	softmax_predictions = softmax_predictions * torch.tensor([1., config.alpha, config.beta]) # set weights.
	sorted_predictions = softmax_predictions


	accs = np.zeros(4)
	rels = np.zeros(4)
	np_matrix = np.zeros((3, 3))
	jet_label = test_data.jet_flavs
	pred = torch.argmax(torch.tensor(sorted_predictions),dim=1) # [20, 50, 30] --argmax-> 1 (c-jet)
	for flav in [0,1,2]:
		correct = len(torch.where(pred[jet_label==flav]==jet_label[jet_label==flav])[0])
		total_label = len(jet_label[jet_label==flav])
		accs[flav] = correct/total_label
	for i in range(len(pred)):
		np_matrix[pred[i], jet_label[i]] += 1
	accs[3] = len(torch.where(pred==jet_label)[0]) / len(jet_label)


	print('{', end='')
	for i in range(3):
		print('{', end='')
		for j in range(3):
			print(int(np_matrix[i][j]), end=', ' if j!=2 else '}')
		print(',' if i!=2 else '}')

	sum_label_arr = np.tile(np_matrix.sum(0), (3, 1))
	print('{', end='')
	for i in range(3):
		print('{', end='')
		for j in range(3):
			print(np_matrix[i][j] / sum_label_arr[j][j], end=', ' if j!=2 else '}')
		print(',' if i!=2 else '}')

	rate_label_arr = np_matrix/sum_label_arr
	rate_label_arr = rate_label_arr.T
	print('{', end='')
	for i in range(3):
		print(rate_label_arr[i][i], end=', ')
		accs[i] = rate_label_arr[i][i]
	print(np.average([rate_label_arr[i][i] for i in range(3)]), end='}\n')

	sum_pred_arr = np.tile(np_matrix.sum(1), (3, 1))
	rate_pred_arr = np_matrix/sum_pred_arr
	print('{', end='')
	for i in range(3):
		print(rate_pred_arr[i][i], end=', ')
		rels[i] = rate_pred_arr[i][i]
	print(np.average([rate_pred_arr[i][i] for i in range(3)]), end='}\n')

	f1 = []
	for flav in range(3):
		f1.append(F1(np_matrix, flav))
	print('{', end='')
	for i in range(3):
		print(f1[i], end=', ')
	print(np.average([f1[i] for i in range(3)]), end='}\n')

	#df = pd.DataFrame(columns=['prediction'])
	#df['prediction'] =  sorted_predictions
	#df.to_csv(config.outputfilename+'_predictions.csv', index=False)

	# ALICE MC jet classification eval
	jet_pt_arr=[]
	for i in range(test_data.n_jets):
		pt_ft_list = (75.95093, 49.134453) # var for detransform (mean, stdev of paper data), from jets_loader.py
		jet_pt_arr.append(test_data.__getitem__(i)[-2][0] * pt_ft_list[1] + pt_ft_list[0])
	pt_bins = [10., 15., 20., 25., 30., 35., 40., 45., 50., 60., 70., 85., 100.]
	file = uproot.recreate("eval_jc.root")
	file['hist_pt_label-b'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array(jet_label==0, dtype="float32"))
	file['hist_pt_pred-b'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array(pred==0, dtype="float32"))
	file['hist_pt_correct-b'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array((jet_label==0) * (pred==0), dtype="float32"))
	file['hist_pt_label-c'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array(jet_label==1, dtype="float32"))
	file['hist_pt_pred-c'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array(pred==1, dtype="float32"))
	file['hist_pt_correct-c'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array((jet_label==1) * (pred==1), dtype="float32"))
	file['hist_pt_label-l'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array(jet_label==2, dtype="float32"))
	file['hist_pt_pred-l'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array(pred==2, dtype="float32"))
	file['hist_pt_correct-l'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array((jet_label==2) * (pred==2), dtype="float32"))
	file['hist_pt_label-b_pred-c'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array((jet_label==0) * (pred==1), dtype="float32"))
	file['hist_pt_label-b_pred-l'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array((jet_label==0) * (pred==2), dtype="float32"))
	file['hist_pt_label-c_pred-b'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array((jet_label==1) * (pred==0), dtype="float32"))
	file['hist_pt_label-c_pred-l'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array((jet_label==1) * (pred==2), dtype="float32"))
	file['hist_pt_label-l_pred-b'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array((jet_label==2) * (pred==0), dtype="float32"))
	file['hist_pt_label-l_pred-c'] = np.histogram(np.array(jet_pt_arr), bins=pt_bins, weights=np.array((jet_label==2) * (pred==1), dtype="float32"))
	
	file['tree'] = uproot.newtree({'jet_flav':'int', 'jet_pt':'float32', 'softmax-b':'float32', 'softmax-c':'float32', 'softmax-l':'float32'})
	file['tree'].extend({'jet_flav':np.array(jet_label), 'jet_pt':np.array(jet_pt_arr), 'softmax-b':np.array(softmax_predictions[:, 0]), 'softmax-c':np.array(softmax_predictions[:, 1]), 'softmax-l':np.array(softmax_predictions[:, 2])})

	file.close()

if __name__ == '__main__':
	main()
