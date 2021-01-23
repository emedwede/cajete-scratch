#!/bin/bash

import numpy as np

#uses our simple partion function
def locatePoint(x):
    if(x < 3):
        return 0
    elif(x >= 3 and x < 6):
        return 1
    else:
        return 2

def main():
    print("\nParticles are binned by partiton blocks {x < 3} | {3 <= x < 6} | {x < 6}\n")
    particles = np.arange(0, 12)
    print("Unshuffled Particles: {}\n".format(particles))
    np.random.shuffle(particles)
    print("Shuffled Particles(ids top, elements bottom):\n{}\n{}\n".format(np.arange(0,12), particles))

    n_blocks = 3

    counts = np.zeros(n_blocks, int)
    offsets = np.zeros(n_blocks, int)
    cell_list = np.zeros(len(particles), int)
    for point in particles:
        counts[locatePoint(point)] += 1

    sum = 0
    for idx in range(0, len(counts)):
        offsets[idx] = sum
        sum += counts[idx]
    print("Counts: {}".format(counts))
    print("Offsets: {}".format(offsets))

    #reset counts to zero
    counts = np.zeros(n_blocks, int)
    for idx in range(0, len(particles)):
        loc = locatePoint(particles[idx])
        cell_list[offsets[loc] + counts[loc]] = idx
        counts[loc] += 1
    print("\nCell List: {}".format(cell_list))

if __name__ == '__main__':
    main()
