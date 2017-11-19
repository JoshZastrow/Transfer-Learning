import sys
import argparse
import os
import pandas as pd


def parse_cameras(infile):

    cameras = ['left', 'center', 'right']

    df = pd.read_csv(infile)

    for cam in cameras:

        # Filter by frame_id then copy that dataframe to a csv file
        if 'frame_id' in df:
            df_filtered = df[df['frame_id'] == '{}_camera'.format(cam)]
            df_filtered.to_csv('{}-interpolated.csv'.format(cam), sep=',')
        else:
            raise KeyError('column frame_id does not exist in this file')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Seperate Camera Indexes.')
    parser.add_argument('infile', metavar='infile', type=str, nargs=1,
                        help='input csv file path')

    args = parser.parse_args()

    parse_cameras(args.infile[0])
