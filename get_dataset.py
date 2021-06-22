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

NEW_TRAINING_DATA_FILE = 'new_train.p'
NEW_TESTING_DATA_FILE = 'new_test.p'
NEW_VALIDATION_DATA_FILE = 'new_val.p'

 
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

 
def load_data(split_load, actions_dict, GT_folder, DATA_folder, segment_file, train=True):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]

    # train_breakfast_data = open(NEW_TRAINING_DATA_FILE, 'wb')
    # val_breakfast_data = open(NEW_VALIDATION_DATA_FILE, 'wb')

    test_breakfast_data = open(NEW_TESTING_DATA_FILE, 'wb')
    if train == True:
        content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

        # if train==True:
        #     train_breakfast_data = open(RAW_TRAINING_DATA_FILE, 'wb')
        # else:
        #     train_breakfast_data = open(TESTING_DATA, 'wb')

        segments = open(segment_file, 'r')
        segment_values = segments.read().split('\n')[:-1]

        train_frames, val_frames, train_segment, val_segment = train_test_split(content_all, segment_values, test_size=0.2, random_state=1)
        
        ## Since tasks are not required as such for our problem
        # all_tasks = ['tea', 'cereals', 'coffee', 'friedegg', 'juice', 'milk', 'sandwich', 'scrambledegg', 'pancake', 'salat']

        train_data_breakfast = []
        train_labels_breakfast = []
        train_segments_breakfast = []

        val_data_breakfast = []
        val_labels_breakfast = []
        val_segments_breakfast = []
        # tasks_breakfast = []
        for (idx, content) in enumerate(train_frames):
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
      
            train_data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))
            train_labels_breakfast.append(torch.tensor(label_curr_video))


            ## for getting the data in the form (segment, label)
            curr_segment = train_segment[idx].split()
            train_segments_breakfast.append(curr_segment)

            pickle.dump((torch.tensor(curr_data, dtype=torch.float64), torch.tensor(label_curr_video), curr_segment), train_breakfast_data)

            print(f'[{idx}] {content} contents dumped')

        print('!!!! GETTING THE VALIDATION DATASET !!!!')
        for (idx, content) in enumerate(val_frames):
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
      
            val_data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))
            val_labels_breakfast.append(torch.tensor(label_curr_video))

            ## for getting the data in the form (segment, label)
            curr_segment = val_segment[idx].split()
            val_segments_breakfast.append(curr_segment)

            pickle.dump((torch.tensor(curr_data, dtype=torch.float64), torch.tensor(label_curr_video), curr_segment), val_breakfast_data)

            print(f'[{idx}] {content} contents dumped')


        #return ((train_data_breakfast, train_labels_breakfast, train_segments_breakfast), (val_data_breakfast, val_labels_breakfast, val_segments_breakfast))


    else:
        content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

        # if train==True:
        #     train_breakfast_data = open(RAW_TRAINING_DATA_FILE, 'wb')
        # else:
        #     train_breakfast_data = open(TESTING_DATA, 'wb')

        segments = open(segment_file, 'r')
        segment_values = segments.read().split('\n')[:-1]

        # train_frames, val_frames, train_segment, val_segment = train_test_split(content_all, segment_values, test_size=0.2, random_state=1)
        
        ## Since tasks are not required as such for our problem
        # all_tasks = ['tea', 'cereals', 'coffee', 'friedegg', 'juice', 'milk', 'sandwich', 'scrambledegg', 'pancake', 'salat']

        test_data_breakfast = []
        test_labels_breakfast = []
        test_segments_breakfast = []
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
      
            test_data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))
            test_labels_breakfast.append(torch.tensor(label_curr_video))


            ## for getting the data in the form (segment, label)
            curr_segment = segment_values[idx].split()
            test_segments_breakfast.append(curr_segment)

            pickle.dump((torch.tensor(curr_data, dtype=torch.float64), torch.tensor(label_curr_video), curr_segment), test_breakfast_data)

            print(f'[{idx}] {content} contents dumped')

        #return test_data_breakfast, test_labels_breakfast, test_segments_breakfast



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
    train = False
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
        load_data(train_split, actions_dict, GT_folder, DATA_folder, train_segment, train=True)
        # print('TRAINING DATA CREATED \n')
        # print('CREATING VALIDATION DATA \n')
        # create_validation_data()
        # print('CREATING SORTED TRAINING DATA \n')
        # get_sorted_data()
    else:
        # print('TEST DATA CREATION \n')
        load_data(test_split, actions_dict, GT_folder, DATA_folder, test_segment, train=False)
    #print('total number videos ' +  str(len(data_labels))  )
