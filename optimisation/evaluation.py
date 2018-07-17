from sklearn.metrics import average_precision_score

class Evaluation:
    def exactSetMatches(self, gold, test):
        gold = set(gold)
        count = 0
        for t in test:
            if t in gold:
                count += 1

        return count/len(test)

    def avgPrecisionScore(self, gold, test):
        return average_precision_score(gold, test)
