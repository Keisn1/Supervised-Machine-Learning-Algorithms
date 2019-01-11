import numpy as np
import matplotlib.pyplot as plt

DICE_PROBABILITIES = [[0.4, 0.3, 0.3], [0.1, 0.9, 0.]]
# choose the dice with best first class


class dice:
    def __init__(self, p_sides):
        self.num_of_sides = len(p_sides)
        self.p_sides = p_sides
        self.alpha = np.ones(self.num_of_sides)*self.num_of_sides
        self.times_rolled = 0

    def roll_dice(self):
        side = np.argmax(np.random.multinomial(1, self.p_sides))
        return side

    def sample_with_current_alpha(self):
        return np.random.dirichlet(self.alpha)


def experiment(p_sides, N):
    dices = [dice(p) for p in DICE_PROBABILITIES]

    for i in range(N):
        sample = np.array([d.sample_with_current_alpha() for d in dices])
        chosen_dice = dices[np.argmax(sample, axis=0)[0]]
        k = chosen_dice.roll_dice()
        chosen_dice.times_rolled += 1
        chosen_dice.alpha[k] += 1

    print np.array([d.times_rolled for d in dices])/float(N)


experiment(DICE_PROBABILITIES, 100)
