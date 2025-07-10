import pandas as pd
import csv
import os
import shutil

def load_labels(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        return [{
            headers[column_index]: row[column_index]
            for column_index in range(len(row))
        }
            for row in reader]


if __name__ == '__main__':
    root = 'kaokore/kaokore'
    data_split =  ['train', 'dev', 'test']
    lbls =  ['gender', 'status']

    categs ={'gender': ['male', 'female'] , 'status': ['noble', 'warrior', 'incarnation', 'commoner']}



    labels = load_labels(os.path.join(root, 'labels.csv'))
    print('Generating imagenet style dataset')
    for lbl in lbls:
        print(lbl)
        os.makedirs('kaokore_imagenet_style/'+lbl, exist_ok=True)
        current_path = 'kaokore_imagenet_style/'+lbl+'/'

        for split in data_split:
            os.makedirs(current_path+split, exist_ok=True)
            for c in categs[lbl]:
                print(c)
                os.makedirs(current_path + split+'/'+c, exist_ok=True)
        for label_entry in labels:
            if os.path.exists(os.path.join(root, 'images_256', label_entry['image'])):
                shutil.copy(os.path.join(root, 'images_256', label_entry['image']),
                            os.path.join(current_path, label_entry['set'], categs[lbl][int(label_entry[lbl])])
                            )
        print('One type of category finished')


