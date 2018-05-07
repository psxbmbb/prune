from xmudata.DIV2K2018 import DIV2K2018
#from xmudata.adddata import AddData
import argparse
#from xmumodel.cyclesr import CycleSR
from xmumodel.edsr_deconv import EDSR
from xmumodel.edsr_deconv_nv_prune import EDSR_P
import tensorflow as tf
import sys
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

FLAGS=None

def main(_):
    data = DIV2K2018(FLAGS.groundtruthdir, FLAGS.datadir, None, None,
                     FLAGS.imgsize, FLAGS.scale, FLAGS.postfixlen, FLAGS.postfixlen)
    
    if(os.path.exists(FLAGS.prunedlist_path)):
        prunedlist = np.loadtxt(FLAGS.prunedlist_path,dtype=np.int64)
    else:
        prunedlist = [0]*19  
    
    network = EDSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale,FLAGS.channels, FLAGS.channels, prunedlist)
    
    network.buildModel()
    network.set_data(data)
    network.train(FLAGS.batchsize, FLAGS.iterations, FLAGS.lr_init, FLAGS.lr_decay, FLAGS.decay_every,
                  FLAGS.savedir, FLAGS.reuse, FLAGS.reusedir, FLAGS.model_step, FLAGS.logdir, FLAGS.prune, FLAGS.prunesize, FLAGS.prune_method, False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--groundtruthdir",default="data/cele/celeba")
    """
    datadir                               postfix_len scale  track
    data/DIV2K_2018/DIV2K_train_LR_x8          2        8    1: bicubic downscaling x8 competition
    data/DIV2K_2018/DIV2K_train_LR_mild        3        4    2: realistic downscaling x4 with mild conditions competition
    data/DIV2K_2018/DIV2K_train_LR_difficult   3        4    3: realistic downscaling x4 with difficult conditions competition
    data/DIV2K_2018/DIV2K_train_LR_wild        4        4    4: wild downscaling x4 competition
    """
    parser.add_argument("--datadir",default="data/cele/celeba_x8")
    parser.add_argument("--valid_groundtruthdir", default='data/cele/valid')
    parser.add_argument("--valid_datadir", default="data/cele/valid_x8")
    #parser.add_argument("--groundtruthdir",default="data/DIV2K_2018/DIV2K_train_HR")
    #parser.add_argument("--datadir",default="data/DIV2K_2018/DIV2K_train_LR_x8")
    #parser.add_argument("--valid_groundtruthdir", default='data/DIV2K/DIV2K_valid_HR')
    #parser.add_argument("--valid_datadir", default="data/DIV2K_2018/DIV2K_valid_LR_x8")

    parser.add_argument("--postfixlen",default=0)
    parser.add_argument("--imgsize",default=16,type=int)
    parser.add_argument("--scale",default=8,type=int)
    parser.add_argument("--layers",default=16,type=int)
    parser.add_argument("--featuresize",default=128,type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--batchsize",default=32,type=int)
    parser.add_argument("--savedir",default='prune_ckpt/tmp')
    parser.add_argument("--logdir", default='log/tmp')
    parser.add_argument("--reusedir",default='ckpt_cele_2')
    parser.add_argument("--psnrpath", default='out/psnr_v6000_train.csv')
    parser.add_argument("--iterations",default=0,type=int)
    parser.add_argument("--lr_init", default=1e-4)
    parser.add_argument("--lr_decay", default=0.5)
    parser.add_argument("--decay_every", default=50000)
    parser.add_argument("--model_step", default=298000)
    parser.add_argument("--reuse", default=True)

    parser.add_argument("--prunedlist_path", default="prune_ckpt/ce_la_pr/v1_124/prunedlistnn")
    parser.add_argument("--prunesize",type=int, default=4)
    parser.add_argument("--prune",default=False,type=bool)
    parser.add_argument("--prune_method",default='thinet')#taylor or thinet
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
