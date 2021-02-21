import math
import numpy as np
import scipy.stats


def get_wins(scores):
    # returns a tuple of win counts (p1 wins, p2 wins)
    scores = np.array(scores)
    return np.sum(scores[:, 0] > scores[:, 1]), np.sum(scores[:, 1] > scores[:, 0])


def get_mean_and_ci(p1_wins, p2_wins):
    z = 1.96  # 95% confidence z-score
    n = p1_wins + p2_wins
    p = p1_wins / n
    interval = z * math.sqrt((p * (1 - p) / n))
    return p, interval


def get_los_from_scores(scores):
    p1_wins, p2_wins = get_wins(scores)
    return get_los(p1_wins, p2_wins)


def get_los(p1_wins, p2_wins):
    # calculate likelihood of superiority for player 1 based on win counts
    # the LOS for player 2 is the complement
    if p1_wins == 0:
        return 0
    if p2_wins == 0:
        return 1

    return scipy.stats.beta(p1_wins, p2_wins).sf(0.5)


def print_inline_stats(scores):
    p1_wins, p2_wins = get_wins(scores)
    p, interval = get_mean_and_ci(p1_wins, p2_wins)

    print(f"Results after {len(scores)} games: {p1_wins}-{p2_wins}    P(player_1): {p:.2f}±{interval:.2f}", end='\r')


def print_stats(scores):
    scores = np.array(scores)

    p1_mean, p2_mean = np.average(scores, axis=0)
    p1_wins, p2_wins = get_wins(scores)

    p1_los = get_los(p1_wins, p2_wins)
    p2_los = 1 - p1_los

    print(f"Wins: {p1_wins:5} {p2_wins:5}")
    print(f"Mean: {p1_mean:.1f} {p2_mean:.1f}")
    print(f"LOS:  {p1_los:0.3f} {p2_los:.3f}")