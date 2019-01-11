#    modelation of click trough rate manager with different approaches
# epsilon_greedy algorithm
# ucb1

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


class dataGenerator:
    def __init__(self, p_keep):
        self.p_keep = p_keep
        self.num_of_cl = len(p_keep)

    def next(self, adv):
        click = 1 if (np.random.random() < self.p_keep[adv]) else 0
        return click


def greedy_eps_experiment(p_keep, N, eps=0.2):
    ''' experiment with greedy_epsilon algorithm
    problem: it will still choose b sometimes

    loss proportianal to N
    '''
    num_of_adv = len(p_keep)
    generator = dataGenerator(p_keep)
    clicked = np.zeros(num_of_adv)
    showed = np.zeros(num_of_adv)
    ctr = np.empty(num_of_adv)

    adv = np.random.randint(0, num_of_adv)
    showed[adv] += 1
    clicked[adv] += generator.next(adv)
    ctr[adv] = clicked[adv]/showed[adv]

    for i in range(1, N):
        if np.random.rand() < eps:
            adv = np.random.randint(0, num_of_adv)
            showed[adv] += 1
            clicked[adv] += generator.next(adv)
            ctr[adv] = clicked[adv]/showed[adv]
        else:
            adv = np.argmax(ctr)
            showed[adv] += 1
            clicked[adv] += generator.next(adv)
            ctr[adv] = clicked[adv]/showed[adv]

    print "greedy_eps"
    print ctr
    print showed


def experiment_ucb(p_keep, N):
    ''' Chernoff Hoeffding bound P(mu_hat > mu + eps) <= exp(-2epsN^2) 
    brings us to reformulation of greedy_eps with epsilons for each 
    advertissement

    loss proportianal to logN
    '''
    num_of_adv = len(p_keep)
    generator = dataGenerator(p_keep)
    clicked = np.zeros(num_of_adv)
    showed = np.ones(num_of_adv)
    ctr = np.empty(num_of_adv)

    adv = np.random.randint(0, num_of_adv)
    showed[adv] += 1
    clicked[adv] += generator.next(adv)
    ctr[adv] = clicked[adv]/showed[adv]

    for i in range(1, N):
        adv = np.argmax(ctr + np.sqrt((2*np.log(i))/showed))
        showed[adv] += 1
        clicked[adv] += generator.next(adv)
        ctr[adv] = clicked[adv]/showed[adv]

    print "ucb"
    print ctr
    print showed


def bayesian(p_keep, N):
    ''' Poserior distribution is the same distribution as the priors
    but with other parameters (parameters are calculated through the parameters of the distribution of the priors)

    example:
    Bernoulli and beta
    a_beta = a_bernoulli + np.sum(current_sample_parameter) 
    b_beta = b_bernoulli + N - np.sum(current_sample_parameter) 

    in terms of CTR
    a_beta = a_bernoulli + #clicks
    b_beta = b_bernoulli + #noclicks

    original a and b
    a = 1, b = 1 -> Beta = Uniform = non_informative prior

    Testing:
    Scenario 1:
    after long while we have sharp distributions -> one with higher mean will be choosen more often (for example higher CTR)

    Scenario 2:
    one sharp, the other less, will still 

    Scenario 3:
    both randomly chosen

    =Thompson sampling

    Important fact for treshhold:
    The Probability of one actual parameter being greater as
    the other can be calculated; watch ab_testing course

    other treshholds:
    given mu2 > mu1
    Lossfunction L = max(m_2 - mu_1, 0)
    stop if Expectation_mu1_mu2(L) < treshhold

    click_trough rate converges to highest prior

    regret = loss if only played best bandit
    '''
    num_of_adv = len(p_keep)
    generator = dataGenerator(p_keep)
    clicked = np.zeros(num_of_adv)
    showed = np.zeros(num_of_adv)
    ctr = np.empty(num_of_adv)
    a = np.ones(num_of_adv)
    b = np.ones(num_of_adv)

    for i in range(0, N):
        realisations = np.random.beta(a, b)
        adv = np.argmax(realisations)
        showed[adv] += 1
        clicked[adv] += generator.next(adv)
        ctr[adv] = clicked[adv]/showed[adv]

        a[adv] += clicked[adv]
        b[adv] += showed[adv]-clicked[adv]

    print "Bayesian"
    print ctr
    print showed


def experiment(p_keep, N):
    generator = dataGenerator(p_keep)
    clicked = np.zeros(len(p_keep))
    for i in range(N):
        clicked += generator.next()
        if i % 25 == 0:
            plt.bar([1, 2], clicked, width=0.4)
            plt.show()


greedy_eps_experiment([0.3, 0.2, 0.1, 0.05], 200000)
experiment_ucb([0.3, 0.2, 0.1, 0.05], 200000)
bayesian([0.3, 0.2, 0.1, 0.05], 200000)
