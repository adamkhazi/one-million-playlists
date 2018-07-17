class Evaluation:
    def exactSetMatches(self, gold, test):
        gold = set(gold)
        count = 0
        for t in test:
            if t in gold:
                count += 1

        return count/len(test)

