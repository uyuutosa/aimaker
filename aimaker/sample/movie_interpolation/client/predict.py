# coding: utf-8

import aimaker.predictor.predictor_factory as pf
import aimaker.trainer as t
import argparse

parser = argparse.ArgumentParser(description='Predict interpolation for movie')
parser.add_argument('path',  type=str,
                    help='feeding movie for intepolate. it can be concatenated separated by comma')
parser.add_argument('predictor', type=str,
                    help='iterp(interpolation)|sr(super resolution) are supported')

parser.add_argument('-o', '--output', nargs='?', default='dump.mp4',
                    help='output path')
parser.add_argument('-d', '--division', nargs='?', default='1', type=int,
                    help='number of division of image')
parser.add_argument('-m', '--mode', nargs='?', default='default',
                    help='interpolation   : default, slash, double\n' +
                         'super resolution: default, concat'
                    )

args = parser.parse_args()

# for camera
if len(args.path) == 1:
    path = int(args.path)
else:
    path = args.path

a = t.trainer()
b = pf.PredictorFactory().create(args.predictor)(a, 
                                                 path, 
                                                 args.division)
b.predict(args.output, mode=args.mode)
