from sklearn.metrics import average_precision_score, recall_score, f1_score, precision_score
from polara.evaluation import evaluation_engine as ee
#from sklearn.metrics import ndcg_score

import sys
import os
sys.path.append(os.path.abspath('../'))
from api import API
from data import Data

class Evaluation:
    def exactSetMatches(self, gold, test):
        gold = set(gold)
        count = 0
        for t in test:
            if t in gold:
                count += 1
        return count/len(test)

    def topK(self, gold, test):
        gold = set(gold)
        count = 0
        for t in test:
            if t in gold:
                count += 1
        return count/len(gold)

    #def NDCG(self, gold, test, k):
        #return ndcg_score(gold, test, k)

    def avgPrecisionScore(self, gold, test):
        return average_precision_score(gold, test)
    
    def recallScore(self, gold, test):
        return recall_score(gold, test)

    def precisionScore(self, gold, test):
        return precision_score(gold, test)

    def f1Score(self, gold, test):
        return f1_score(gold, test)
