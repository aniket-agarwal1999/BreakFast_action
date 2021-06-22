import torch
import torch.nn as nn
import numpy as np
from model import *
from torch import optim
import pickle
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil
from datetime import datetime
import time
import os


TRAINING_DATA = 'sorted_training_data.p'
VALIDATION_DATA = 'validation_data.p'

RESULT_DIR = 'Old_Outputs/'

def transform_input(segments):
    ## For padding
    input_dim = 400 # num features in i3D frame
    segment_lengths = [len(segments) for segments in segments]
    
    # empty matrix with zero padding
    longest_segment = max(segment_lengths)
    batch_size = len(segments)
    padded_segments = np.zeros((batch_size, longest_segment, input_dim))
    
    for i, length in enumerate(segment_lengths):
        sequence = segments[i]
        padded_segments[i, 0:length] = sequence[:length]
    
    transformed_segments = []
    for padded_segment in padded_segments:
        transformed_segments.append(torch.Tensor(padded_segment).float())
    
    return transformed_segments

def get_next_data_batch(f, batch_size):    
    raw_segments = []
    labels = []
    
    is_end_reached = False
    
    num_segments = 0
    while num_segments < batch_size:
        try:
            (segment, label) = pickle.load(f)
            
            raw_segments.append(segment)
            labels.append(label)
            num_segments += 1
        except (EOFError):
            is_end_reached = True
            raw_segments = []
            labels = []
            break
    
    if len(raw_segments) == 0:
        return [], [], is_end_reached
    
    padded_segments = transform_input(raw_segments)
    segments = torch.stack(padded_segments).float()
    labels = torch.Tensor(labels).long()
    
    return segments, labels, is_end_reached

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def save_checkpoint(state, is_best, folder):
    filename = folder + 'model_checkpoint.pth'
    torch.save(state, filename)
    
    if is_best:
        shutil.copyfile(filename, folder + 'model_best.pth')


def train(model, batch_size, num_epochs, train_data_file = TRAINING_DATA, validation_file = VALIDATION_DATA):

	# model = model.to('cuda')
	training_loss = []
	validation_loss = []

	best = float("inf")

	print('!! TRAINING !!')
	for epoch in range(num_epochs):

		train_f = open(train_data_file, 'rb')
		epoch_end = False

		iteration = 0
		epoch_training_losses = []

		while not epoch_end:
			model.train()

			segments, labels, epoch_end = get_next_data_batch(train_f, batch_size)
			if len(segments) == 0:
				break

			segments, labels = segments.to('cuda'), labels.to('cuda')
			segments = segments.requires_grad_()

			### To avoid the retain_graph issue
			# hidden = repackage_hidden(hidden)
			optimizer.zero_grad()

			# print('!! Shape: !! ', segments.shape)
			outputs = model(segments)

			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			epoch_training_losses.append(loss.item())

			iteration += 1
			if iteration%50 == 0:
				print('Epoch: ', epoch, ' iter: ', iteration, ' Loss: ', loss)


		train_f.close()

		val_f = open(validation_file, 'rb')
		model.eval()

		correct = 0
		total = 0
		end_reach = False

		epoch_validation_losses = []

		while not end_reach:
			segments, labels, end_reach = get_next_data_batch(val_f, 16)
			if len(segments) == 0:
				break
			segments, labels = segments.to('cuda'), labels.to('cuda')
        
			outputs = model(segments)

			loss = criterion(outputs, labels)
			epoch_validation_losses.append(loss.item())

			_, predicted = torch.max(outputs.data, 1)

			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		val_f.close()

		accuracy = 100 * correct / total

		training_loss.append(np.mean(epoch_training_losses))
		validation_loss.append(np.mean(epoch_validation_losses))

		print('!!! VALIDATION LOSS: !!! ', validation_loss)
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

	batch_size = 16
	input_dim = 400
	n_layers = 2
	hidden_dim = 160
	classes = 48
	epochs = 15

	model_time = str(datetime.now().strftime("%Y-%m-%d___%H-%M-%S"))
	os.mkdir(RESULT_DIR + model_time)
	print(model_time)
	shutil.copyfile('train.py', RESULT_DIR + model_time + '/' + model_time + '_train.py')
	shutil.copyfile('model.py', RESULT_DIR + model_time + '/' + model_time + '_model.py')

	model = BiLSTM_model(input_dim, hidden_dim, n_layers, classes)

	criterion = nn.CrossEntropyLoss()
	# lr = 0.01
	# optimizer = optim.Adam(model.parameters(), lr= lr)

	model = model.to('cuda')

	## Optimizer
	learning_rate = 0.01
	weight_decay = 0.00005
	# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	# hidden = (torch.zeros(n_layers*2, batch_size, hidden_dim).to('cuda'), torch.zeros(n_layers*2, batch_size, hidden_dim).to('cuda'))

	training_loss, validation_loss = train(model, batch_size, epochs)




