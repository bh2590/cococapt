from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.eval_inmemory import COCOEvalCap as COCOEvalCap2
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

def get_scores(annFile='./evaluate/annotations/captions_val2017.json',
         resFile='./evaluate/results/captions_val2017_fakecap_results.json',
         evalImgsFile=None, evalFile=None):
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    scores = {}
    for metric, score in cocoEval.eval.items():
        scores[metric] = score
        print('%s: %.3f' % (metric, score))

    if evalImgsFile:
        json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
    if evalFile:
        json.dump(cocoEval.eval, open(evalFile, 'w'))

    return scores


def get_scores_im(result_list, annFile='./evaluate/annotations/captions_val2017.json',
          evalImgsFile=None, evalFile=None):
    coco = COCO(annFile)
    cocoEval = COCOEvalCap2(coco)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate(result_list)
    scores = {}
    for metric, score in cocoEval.eval.items():
        scores[metric] = score
        print('%s: %.3f' % (metric, score))

    if evalImgsFile:
        json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
    if evalFile:
        json.dump(cocoEval.eval, open(evalFile, 'w'))

    return scores