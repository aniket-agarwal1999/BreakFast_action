import torch.nn as nn
import numpy as np
from model import *
from torch import optim
import pickle
from torch.autograd import Variable
import torch
import sklearn
from sklearn.model_selection import train_test_split
from get_dataset import load_data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil
from datetime import datetime
import time
import os


# train_split =  'splits/train.split1.bundle'
# test_split  =  'splits/test.split1.bundle'
# GT_folder   =  'groundTruth/'
# DATA_folder =  'data/'
# mapping_loc =  'splits/mapping_bf.txt'
# train_segment = 'train_segments.txt'
# test_segment = 'test_segments.txt'
RESULT_DIR = 'Outputs/'

TRAINING_DATA_COUNT = 1168
VAL_DATA_COUNT = 292

NEW_TRAINING_DATA_FILE = 'new_train.p'
NEW_TESTING_DATA_FILE = 'new_test.p'
NEW_VALIDATION_DATA_FILE = 'new_val.p'

def read_mapping_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]

    actions_dict=dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

actions_dict = read_mapping_dict(mapping_loc)

def save_checkpoint(state, is_best, folder):
    filename = folder + 'model_checkpoint.pth'
    torch.save(state, filename)
    
    if is_best:
        shutil.copyfile(filename, folder + 'model_best.pth')


def train(model, batch_size, num_epochs):

	# model = model.to('cuda')

	training_loss = []
	validation_loss = []

	train_f = open(NEW_TRAINING_DATA_FILE, 'rb')
	val_f = open(NEW_VALIDATION_DATA_FILE, 'rb')

	train_data_breakfast = []
	train_labels_breakfast = []
	train_segments_breakfast = []

	val_data_breakfast = []
	val_labels_breakfast = []
	val_segments_breakfast = []


	#((train_data_breakfast, train_labels_breakfast, train_segments_breakfast), (val_data_breakfast, val_labels_breakfast, val_segments_breakfast)) = load_data(train_split, actions_dict, GT_folder, DATA_folder, train_segment, train=True)

	counter = 1
	while True:
		try:
			data, label, segment = pickle.load(train_f)
			if counter % 100 == 0:
				print(f"at sample: {counter}")

			train_data_breakfast.append(data)
			train_labels_breakfast.append(label)
			train_segments_breakfast.append(segment)
			counter += 1
		except (EOFError):
			break
	
	train_f.close()

	counter = 1
	while True:
		try:
			data, label, segment = pickle.load(val_f)
			if counter % 100 == 0:
				print(f"at sample: {counter}")

			val_data_breakfast.append(data)
			val_labels_breakfast.append(label)
			val_segments_breakfast.append(segment)
			counter += 1
		except (EOFError):
			break

	val_f.close()

	best = float("inf")

	print('!!! TRAINING !!!')
	for epoch in range(num_epochs):

		epoch_training_loss = []

		for idx in range(TRAINING_DATA_COUNT):
			model.train()

			frames, labels, segments = train_data_breakfast[idx], train_labels_breakfast[idx], train_segments_breakfast[idx]

			frames = frames.unsqueeze(0)

			frames, labels = frames.to('cuda'), labels.to('cuda')
			frames = frames.requires_grad_()

			optimizer.zero_grad()

			outputs, labels = model(frames, segments, labels)

			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			epoch_training_loss.append(loss.item())

			if idx % 50 == 0:
				print('Epoch: ', epoch, ' iter: ', idx, ' Loss: ', loss.item())


		model.eval()

		correct = 0
		total = 0

		epoch_validation_losses = []

		for ind in range(VAL_DATA_COUNT):
			frames, labels, segments = val_data_breakfast[ind], val_labels_breakfast[ind], val_segments_breakfast[ind]

			frames, labels = frames.to('cuda'), labels.to('cuda')

			frames = frames.unsqueeze(0)

			outputs, labels = model(frames, segments, labels)

			loss = criterion(outputs, labels)
			epoch_validation_losses.append(loss.item())

			_, predicted = torch.max(outputs.data, 1)

			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		accuracy = 100 * correct / total

		training_loss.append(np.mean(epoch_training_loss))
		validation_loss.append(np.mean(epoch_validation_losses))

		scheduler.step(validation_loss[-1])

		print('!!! VALIDATION LOSS: !!! ', np.mean(epoch_validation_losses))
		print('!!! VALIDATION ACCURACY: !!!', accuracy)

		is_best = validation_loss[-1] < best
		best = min(validation_loss[-1], best)

		save_checkpoint({
			'next_epoch_idx': epoch + 1,
			'state_dict': model.state_dict(),
			'val_loss': validation_loss[-1],
			'optimizer': optimizer.state_dict()
		}, is_best, RESULT_DIR + model_time+'/')


	return training_loss, validation_loss


if __name__ == "__main__":

	### Hyperparameters
	batch_size = 1
	input_dim = 400
	n_layers = 3
	hidden_dim = 160
	segment_hidden_dim = 100
	classes = 48
	epochs = 30

	model_time = str(datetime.now().strftime("%Y-%m-%d___%H-%M-%S"))
	os.mkdir(RESULT_DIR + model_time)
	print(model_time)
	shutil.copyfile('new_train.py', RESULT_DIR + model_time + '/' + model_time + '_new_train.py')
	shutil.copyfile('model.py', RESULT_DIR + model_time + '/' + model_time + '_model.py')

	model = new_model(input_dim, hidden_dim, segment_hidden_dim, n_layers, classes)
	model = model.to('cuda')

	criterion = nn.CrossEntropyLoss()
	# lr = 0.01
	# optimizer = optim.Adam(model.parameters(), lr= lr)

	## Optimizer
	learning_rate = 0.01
	weight_decay = 0.00005
	momentum = 0.9
	#optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

	patience = 2
	decrease_factor = 0.7
	min_learning_rate = 0.00005
	scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                              patience=patience, min_lr=min_learning_rate, factor=decrease_factor,
                              verbose=True)

	# hidden = (torch.zeros(n_layers*2, batch_size, hidden_dim).to('cuda'), torch.zeros(n_layers*2, batch_size, hidden_dim).to('cuda'))

	training_loss, validation_loss = train(model, batch_size, epochs)

