import torch
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

TEST_DATA_COUNT = 252

NEW_TESTING_DATA_FILE = 'new_test.p'
RESULT_DIR = 'Outputs/'
CKPT_DIR = '2021-01-18___15-57-26/'

def save_checkpoint(state, is_best, folder):
    filename = folder + 'model_checkpoint.pth'
    torch.save(state, filename)
    
    if is_best:
        shutil.copyfile(filename, folder + 'model_best.pth')

def test(model, batch_size):

	test_f = open(NEW_TESTING_DATA_FILE, 'rb')

	test_data_breakfast = []
	test_labels_breakfast = []
	test_segments_breakfast = []

	counter = 1
	while True:
		try:
			data, label, segment = pickle.load(test_f)
			if counter % 100 == 0:
				print(f"at sample: {counter}")

			test_data_breakfast.append(data)
			test_labels_breakfast.append(label)
			test_segments_breakfast.append(segment)
			counter += 1
		except (EOFError):
			break

	test_f.close()

	test_loss = []
	model.eval()

	correct = 0
	total = 0

	print('!!! TESTING !!!')
	for idx in range(TEST_DATA_COUNT):
		frames, labels, segments = test_data_breakfast[idx], test_labels_breakfast[idx], test_segments_breakfast[idx]

		frames, labels = frames.to('cuda'), labels.to('cuda')
		frames = frames.unsqueeze(0)

		outputs, labels = model(frames, segments, labels)

		loss = criterion(outputs, labels)
		test_loss.append(loss.item())

		_, predicted = torch.max(outputs.data, 1)

		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	accuracy = 100 * correct/total

	print('!!! TEST LOSS: !!! ', np.mean(test_loss))
	print('!!! ACCURACY: !!! ', accuracy)

	return np.mean(test_loss)


if __name__ == "__main__":

	batch_size = 1
	input_dim = 400
	n_layers = 2
	hidden_dim = 160
	segment_hidden_dim = 100
	classes = 48

	criterion = nn.CrossEntropyLoss()

	model = new_model(input_dim, hidden_dim, segment_hidden_dim, n_layers, classes)

	if os.path.isdir(RESULT_DIR + CKPT_DIR):
		model = model.to('cuda')
		ckpt = torch.load(RESULT_DIR + CKPT_DIR + 'model_best.pth')
		model.load_state_dict(ckpt['state_dict'])

		test_loss = test(model, batch_size)

	else:
		print('!!!! THE CHECKPOINT DOES NOT EXIST !!!!')
