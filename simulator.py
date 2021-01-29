from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray

import stats

Configuration = namedtuple('Configuration', ['banditCount'])
Observation = namedtuple('Observation', ['step', 'reward', 'agentIndex', 'lastActions'])


def smoke_test(agent):
    config = Configuration(banditCount=100)
    obs = Observation(step=0, reward=0, agentIndex=0, lastActions=[])
    action = agent.step(obs, config)

    obs = Observation(step=1, reward=0, agentIndex=0, lastActions=[action, 2])
    action = agent.step(obs, config)

    obs = Observation(step=2, reward=1, agentIndex=0, lastActions=[action, 5])
    action = agent.step(obs, config)


@ray.remote
def simulate_mab(agent_lambdas, num_steps=2000, num_bandits=100, game_id=0):
    config = Configuration(banditCount=num_bandits)
    probs = np.random.rand(num_bandits)
    last_actions = [0, 0]
    totals = [0, 0]
    agents = [l(num_bandits) for l in agent_lambdas]

    d = {'step': [], 'p1_total': [], 'p2_total': []}

    for i in range(num_steps):
        for j, agent in enumerate(agents):
            obs = Observation(step=i, reward=totals[j], agentIndex=j, lastActions=last_actions)
            choice = agent.step(obs, config)
            totals[j] += np.random.rand() < probs[choice]
            last_actions[j] = choice

        d['step'].append(i)
        d['p1_total'].append(totals[0])
        d['p2_total'].append(totals[1])

        for action in last_actions:
            probs[action] *= 0.97

    df = pd.DataFrame(data=d)
    df['diff'] = df.p1_total - df.p2_total
    df['game_id'] = game_id

    return totals, df


def compare_agents(agent_lambdas, num_games=50, num_bandits=100, num_steps=2000, min_games=20):
    names = [a(num_bandits).description() for a in agent_lambdas]
    print(f"{names[0]}\n{names[1]}")

    num_cpus = 4

    scores = []
    df = pd.DataFrame()

    for i in range(0, num_games, num_cpus):
        result_ids = [simulate_mab.remote(
            agent_lambdas,
            num_steps=num_steps,
            num_bandits=num_bandits,
            game_id=i) for i in range(i, i + num_cpus)]

        batch_results = ray.get(result_ids)

        for score, game_df in batch_results:
            scores.append(score)
            df = df.append(game_df)

        stats.print_inline_stats(scores)
        p1_wins, p2_wins = stats.get_wins(scores)
        p1_los = stats.get_los(p1_wins, p2_wins)
        if len(scores) >= min_games and (p1_los > 0.99 or p1_los < 0.01):
            break

    print('\n')

    return scores, df


def rank_agents(agents, num_games=50, min_games=20):
    num_agents = len(agents)
    if num_agents == 1:
        return agents

    if num_agents == 2:
        scores, _ = compare_agents(agents, num_games=num_games, min_games=min_games)
        p1_wins, p2_wins = stats.get_wins(scores)
        if p1_wins > p2_wins:
            return agents
        else:
            return agents[::-1]

    sublist_1 = rank_agents(agents[:num_agents // 2], num_games=num_games, min_games=min_games)
    sublist_2 = rank_agents(agents[num_agents // 2:], num_games=num_games, min_games=min_games)

    ranked_agents = []
    while True:
        if not sublist_1:
            ranked_agents += sublist_2
            break

        if not sublist_2:
            ranked_agents += sublist_1
            break

        scores, _ = compare_agents([sublist_1[0], sublist_2[0]], num_games=num_games, min_games=min_games)
        p1_wins, p2_wins = stats.get_wins(scores)
        if p1_wins > p2_wins:
            ranked_agents.append(sublist_1.pop(0))
        else:
            ranked_agents.append(sublist_2.pop(0))

    return ranked_agents


def round_robin(agent_lambdas, num_games=50, num_bandits=100, num_steps=2000, min_games=20):
    num_agents = len(agent_lambdas)
    num_rounds = (num_agents * (num_agents - 1)) // 2
    round_num = 1
    agent_names = [l(num_bandits).description() for l in agent_lambdas]
    records = np.zeros((num_agents, 3), dtype='int32')
    los_matrix = np.full((num_agents, num_agents), 0.5)

    print(f"Starting round robin with {num_agents} agents:")
    for name in agent_names:
        print(name)
    print("")

    for i in range(num_agents - 1):
        for j in range(i + 1, num_agents):
            print(f"Starting round {round_num} of {num_rounds}:")
            scores, _ = compare_agents([agent_lambdas[i], agent_lambdas[j]],
                                       num_games=num_games,
                                       num_bandits=num_bandits,
                                       num_steps=num_steps,
                                       min_games=min_games)
            p1_wins, p2_wins = stats.get_wins(scores)
            ties = len(scores) - p1_wins - p2_wins
            records[i] += [p1_wins, p2_wins, ties]
            records[j] += [p2_wins, p1_wins, ties]

            p, _ = stats.get_mean_and_ci(p1_wins, p2_wins)
            los_matrix[i, j] = p
            los_matrix[j, i] = 1 - p

            round_num += 1

    for i in range(num_agents):
        print(f"{agent_names[i]}: {'-'.join(map(str, records[i]))}")

    plot_los_heatmap(agent_names, los_matrix)


def graph_game_results(df):
    ax = df.groupby('step').mean()['diff'].rolling(window=10).mean().plot(
        title="Point difference averages over all games")
    ax.set_xlabel("step")
    ax.set_ylabel("P1 - P2")


def plot_los_heatmap(agent_names, los_matrix):
    num_agents = len(agent_names)

    order = np.argsort(-np.sum(los_matrix, axis=1))
    sorted_names = [agent_names[x] for x in order]
    sorted_los = los_matrix[order][:, order]

    fix, ax = plt.subplots()
    ax.imshow(sorted_los, cmap='gray', vmin=0, vmax=1.5)

    # We want to show all ticks...
    ax.set_xticks(np.arange(num_agents))
    ax.set_yticks(np.arange(num_agents))

    # ... and label them with the respective list entries
    ax.set_xticklabels(sorted_names)
    ax.set_yticklabels(sorted_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            ax.text(j, i, "{:.2f}".format(sorted_los[i, j]), ha="center", va="center", color="w")
    plt.show()
