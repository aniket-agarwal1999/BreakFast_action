import os  
import torch
import numpy as np
import os.path
import pickle
import sklearn
from sklearn.model_selection import train_test_split

RAW_TRAINING_DATA_FILE = 'raw_training_data.p'
UNSORTED_TRAINING_DATA = 'unsorted_training_data.p'
SORTED_TRAINING_DATA = 'sorted_training_data.p'
VALIDATION_DATA = 'validation_data.p'
TESTING_DATA = 'testing_data.p'
 
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

 
def load_data(split_load, actions_dict, GT_folder, DATA_folder, segment_file, train=True):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

    if train==True:
        train_breakfast_data = open(RAW_TRAINING_DATA_FILE, 'wb')
    else:
        train_breakfast_data = open(TESTING_DATA, 'wb')

    segments = open(segment_file, 'r')
    segment_values = segments.read().split('\n')[:-1]
    
    ## Since tasks are not required as such for our problem
    # all_tasks = ['tea', 'cereals', 'coffee', 'friedegg', 'juice', 'milk', 'sandwich', 'scrambledegg', 'pancake', 'salat']

    data_breakfast = []
    labels_breakfast = []
    # tasks_breakfast = []
    for (idx, content) in enumerate(content_all):
        # curr_task = content.split('_')[-1].split('.')[0]
        # tasks_breakfast.append(int( all_tasks.index(curr_task)) )

        file_ptr = open(GT_folder + content, 'r')
        curr_gt = file_ptr.read().split('\n')[:-1]
        label_seq, length_seq = get_label_length_seq(curr_gt)

        loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'
        curr_data = np.loadtxt(loc_curr_data, dtype='float32')

        label_curr_video = []
        for iik in range(len(curr_gt)):
            label_curr_video.append(actions_dict[curr_gt[iik]] )
  
        data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))
        labels_breakfast.append(label_curr_video )

        ## for getting the data in the form (segment, label)
        curr_segment = segment_values[idx].split()
        for i in range(len(curr_segment) - 1):
            start_segment = int(curr_segment[i])
            end_segment = int(curr_segment[i+1])
            curr_segment_frames = curr_data[start_segment: end_segment]
            curr_segment_label = label_curr_video[start_segment]
            pickle.dump((torch.tensor(curr_segment_frames, dtype=torch.float64), curr_segment_label),
                        train_breakfast_data)
    
        print(f'[{idx}] {content} contents dumped')

    return  data_breakfast, labels_breakfast

def create_validation_data():
    f = open(RAW_TRAINING_DATA_FILE, 'rb')

    input_frames = []
    input_labels = []
    counter = 1
    while True:
        try:
            (segment, label) = pickle.load(f)
        
            if counter % 100 == 0:
                print(f"at sample: {counter}")
            input_frames.append(segment)
            input_labels.append(label)
            counter += 1
        except (EOFError):
            break
    
    f.close()

    X_train, X_val, y_train, y_val = train_test_split(input_frames, input_labels, test_size=0.2, random_state=1)

    print(len(X_train))
    print(len(X_val))

    training_out = open(UNSORTED_TRAINING_DATA, 'wb')
    counter = 1
    for i, segment  in enumerate(X_train):
        if counter % 100 == 0:
            print(f"dumping sample {counter} in {training_out}") 
        pickle.dump((segment, y_train[i]), training_out)
        counter += 1
    training_out.close()

    val_out = open(VALIDATION_DATA, 'wb')
    counter = 1
    for i, segment in enumerate(X_val):
        if counter % 100 == 0:
            print(f"dumping sample {counter} in {val_out}")
        pickle.dump((segment, y_val[i]), val_out)
        counter += 1

    val_out.close()

def get_sorted_data():
    f = open(UNSORTED_TRAINING_DATA, 'rb')

    segments = []
    labels = []
    segment_lengths = []

    segment_idx = 0
    while True:
        try:
            (segment, label) = pickle.load(f)

            if segment_idx % 100 == 0:
                print(f"at sample: {segment_idx }")

            segments.append(segment)
            labels.append(label)
            segment_lengths.append((segment_idx , len(segment)))

            segment_idx  += 1

        except (EOFError):
            break

    f.close()

    sorted_segment_lengths = sorted(segment_lengths, key=lambda tup: tup[1])

    # Store sorted training data
    training_out = open(SORTED_TRAINING_DATA, 'wb')

    counter = 1
    for (idx, length) in sorted_segment_lengths:
        if counter % 100 == 0:
            print(f"dumping sample {counter} in {training_out}") 

        pickle.dump((segments[idx], labels[idx]), training_out)
        counter += 1

    training_out.close()


def get_label_bounds( data_labels):
    labels_uniq = []
    labels_uniq_loc = []
    for kki in range(0, len(data_labels) ):
        uniq_group, indc_group = get_label_length_seq(data_labels[kki])
        labels_uniq.append(uniq_group)
        labels_uniq_loc.append(indc_group)
    return labels_uniq, labels_uniq_loc

def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))

    return label_seq, length_seq


def get_maxpool_lstm_data(cData, indices):
    list_data = []
    for kkl in range(len(indices)-1):
        cur_start = indices[kkl]
        cur_end = indices[kkl+1]
        if cur_end > cur_start:
            list_data.append(torch.max(cData[cur_start:cur_end,:],
                                       0)[0].squeeze(0))
        else:
            list_data.append(torch.max(cData[cur_start:cur_end+1,:],
                                       0)[0].squeeze(0))
    list_data  =  torch.stack(list_data)
    return list_data

def read_mapping_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]

    actions_dict=dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

if __name__ == "__main__":
    train = True
    if train:
        split = 'train'
    else:
        split = 'test'
    COMP_PATH = ''
    
    train_split =  os.path.join(COMP_PATH, 'splits/train.split1.bundle')
    test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle')
    GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/')
    DATA_folder =  os.path.join(COMP_PATH, 'data/')
    mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt')
    train_segment = os.path.join(COMP_PATH, 'train_segments.txt')
    test_segment = os.path.join(COMP_PATH, 'test_segments.txt')

  
    actions_dict = read_mapping_dict(mapping_loc)
    if split == 'train':
        data_feat, data_labels = load_data(train_split, actions_dict, GT_folder, DATA_folder, train_segment, train=True)
        print('TRAINING DATA CREATED \n')
        print('CREATING VALIDATION DATA \n')
        create_validation_data()
        print('CREATING SORTED TRAINING DATA \n')
        get_sorted_data()
    else:
        print('TEST DATA CREATION \n')
        data_feat, data_labels = load_data(test_split, actions_dict, GT_folder, DATA_folder, test_segment, train=False)
    print('total number videos ' +  str(len(data_labels))  )
