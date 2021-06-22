import pandas as pd
import glob

split_load = 'splits/test.split1.bundle'
files = open(split_load, 'r')
file_paths = files.read().split('\n')[1:-1]
file_paths = [x.strip('./data/') + 't' for x in file_paths]

train_segments_file = open('test_segments.txt', 'w')
for file_path in file_paths:
    print("Get segment split for file: ", file_path)
    text_file = open(file_path, "r")
    labels = text_file.read().split()
    text_file.close()

    labels_pd = pd.DataFrame({"sub_action": labels})
    segments = (labels_pd != labels_pd.shift()).cumsum()
    segments = segments[labels_pd["sub_action"] != "SIL"]  # ignore "SIL" segments
    segment_split_indices = list(segments.groupby("sub_action").apply(lambda x: (x.index[0], x.index[-1])))
    print(segment_split_indices)

    segment_split = ""
    for (idx, segment_indices) in enumerate(segment_split_indices):
        start = segment_indices[0]
        end = segment_indices[1]

        if idx == 0:
            segment_split += str(start) + " " + str(end)
        else:
            segment_split += " " + str(end)

    print(segment_split)
    train_segments_file.write("%s\n" % segment_split)

print("Finished getting segment splits for train .txt files")
train_segments_file.close()
