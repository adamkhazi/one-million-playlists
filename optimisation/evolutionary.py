#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import json

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import sys
import os
import math

from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath('../'))
from data import Data

import pdb

d = Data()
SET_SIZE = 5
#NR_FEATURES = trackFeatures.shape[1]
NR_FEATURES = 3
FEATURE_INDICES = [0,1,10]
trackFeatures, trackNames = d.getTrackFeaturesWNames( 150000 )
trackFeatures = trackFeatures[:,FEATURE_INDICES]

# features have equal importance
scaler = MinMaxScaler()
trackFeatures = scaler.fit_transform(trackFeatures)

cons = d.getGoldSetAvgCons('Fun Run 150–165 BPM')

#ideal = [0.458, 0.591, 5, -5.621, 1, 0.0326, 0.568, 0.0, 0.286, 0.654, 50.558, 161187, 3]
ideal = [cons[i] for i in FEATURE_INDICES]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("indices", random.sample, range(trackFeatures.shape[0]), SET_SIZE)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalTSP(individual):
    total = [0 for i in range(NR_FEATURES)]
    for gene in individual:
        for i, f in enumerate(trackFeatures[gene]):
            total[i] += math.sqrt((ideal[i]-f)**2)
    return tuple(total)

def feasible(individual):
    uniq = set()
    for gene in individual:
        uniq.add(gene)
    return True if len(uniq) == len(individual) else False

def distance(individual):
    uniq = set()
    for gene in individual:
        uniq.add(gene)
    diff = len(individual) - len(uniq)
    return 10000 * diff

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.4)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evalTSP)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 10000.0, distance))

def main():
    pop = toolbox.population(n=10000)

    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    algorithms.eaSimple(pop, toolbox, 0.7, 0.3, 250, stats=stats, halloffame=hof)


    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()

    print('track names in hof individual set size', SET_SIZE)
    for h in hof:
        for i in h:
            print(trackNames[i])
        print(' === ')

    pdb.set_trace()