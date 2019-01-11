import numpy as np
import matplotlib.pyplot as plt

HUMAN_PARAMS = [0, 1, 2]
# choose the dice with best first class


class human_group:
    def __init__(self, mu):
        self.mu = mu
        self.tau = 1

        self.predicted_mu = 0
        self.predicted_lambda = 1
        # no need to keep m_0 since we say it's zero in the beginning
        self.lambda_0 = 1

        self.sum_of_heights = 0
        self.times_chosen = 0

    def rand_height(self):
        height = np.random.normal(self.mu, self.tau)
        return height

    def sample(self):
        return np.random.normal(self.predicted_mu, 1/np.sqrt(self.predicted_lambda))

    def update(self, height):
        self.times_chosen += 1
        self.sum_of_heights += height

        # parameter updates on assuming that human_params[1] is known
        self.predicted_lambda += self.tau
        self.predicted_mu = (
            self.tau * self.sum_of_heights) / self.predicted_lambda


def experiment(p_sides, N):
    human_groups = [human_group(params) for params in HUMAN_PARAMS]

    for n in range(1, N+1):
        chosen_group_num = np.argmax([group.sample()
                                      for group in human_groups])
        chosen_group = human_groups[chosen_group_num]
        height = chosen_group.rand_height()
        chosen_group.update(height)

    return np.argmax([g.times_chosen for g in human_groups])


for i in range(1000):
    most = experiment(HUMAN_PARAMS, 1000)
    if most != 2:
        print "false"

print "hey"
