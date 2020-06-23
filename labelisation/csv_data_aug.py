import os
import csv
import cv2
from imgaug import augmenters as iaa
from absl import app, flags
import pandas as pd

def main(_):
    flip = iaa.Fliplr()
    df = pd.read_csv(FLAGS.csv_file)
    for index,row in df.iterrows():
        print(index)
        if row['filename'][0]!='f' and df[df['filename']=='flipped_'+row['filename']].shape[0]==0:
            img=cv2.imread('../data/jpg/'+row[0])
            if img.any():
                flipped = flip.augment_image(img)
                new_row = {'filename':'flipped_'+row['filename'],
                            'width':row['width'],
                            'height':row['height'],
                            'class':row['class'],
                            'xmin':int(row['width'])-int(row['xmax']),
                            'ymin':row['ymin'],
                            'xmax':int(row['width'])-int(row['xmin']),
                            'ymax':row['ymax']}
                df = df.append(new_row, ignore_index=True)
                cv2.imwrite('../data/jpg/flipped_' + row['filename'], flipped)
    df.to_csv(FLAGS.csv_file, index=False)
    
if __name__=='__main__':
    FLAGS=flags.FLAGS
    flags.DEFINE_string('csv_file','','path to csv file')
    app.run(main)