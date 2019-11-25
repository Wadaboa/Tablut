'''
Tablut states evaluation functions weights computation
'''

import random
import time
import bisect
from multiprocessing import Process, Array
from datetime import datetime


import tablut_player.utils as utils
import tablut_player.config as conf
import tablut_player.strategy as strat
import tablut_player.heuristic as heu
from tablut_player.game import TablutGame


HEURISTIC_WEIGHTS_RANGE = [0, 20]


class TablutPlayer():

    def __init__(self, weights, white_wins=0, black_wins=0):
        self.weights = weights
        self.white_wins = white_wins
        self.black_wins = black_wins

    def __eq__(self, other):
        for weight_one, weight_two in zip(self.weights, other.weights):
            if weight_one != weight_two:
                return False
        return True

    def __str__(self):
        return (
            f'Weights: {self.weights}\n'
            f'White wins: {self.white_wins}\n'
            f'Black wins: {self.black_wins}\n'
        )

    def __repr__(self):
        return (
            f'Weights: {self.weights}\n'
            f'White wins: {self.white_wins}\n'
            f'Black wins: {self.black_wins}'
        )


def tournament(population):
    results = Array('i', [0] * len(population) * len(population))
    game_num = 0
    processes = []
    for player_one in population:
        for player_two in population:
            p = Process(
                target=play,
                args=(player_one, player_two, results, game_num)
            )
            p.start()
            processes.append(p)
            game_num += 1
    time.sleep(3)
    for p in processes:
        p.join()
    game_num = 0
    for player_one in population:
        for player_two in population:
            if results[game_num] == 1:
                player_one.white_wins += 1
            elif results[game_num] == -1:
                player_two.black_wins += 1
            game_num += 1
    return population


def play(player_one, player_two, results, game_num, max_turns=50):
    game = TablutGame()
    game_state = game.initial
    black_ttable = strat.TT()
    white_ttable = strat.TT()
    while not game.terminal_test(game_state) and game.turn < max_turns:
        if game.turn % 10 == 0:
            black_ttable.clear()
            white_ttable.clear()
        game.inc_turn()
        heu.set_heuristic_weights(player_one.weights)
        white_move = strat.get_move(
            game, game_state, conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD,
            tt=white_ttable
        )
        game_state = game.result(game_state, white_move)
        if game.terminal_test(game_state):
            results[game_num] = 1
            player_one.white_wins += 1
            break
        heu.set_heuristic_weights(player_two.weights)
        black_move = strat.get_move(
            game, game_state, conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD,
            prev_move=white_move, tt=black_ttable
        )
        game_state = game.result(game_state, black_move)
        if game.terminal_test(game_state):
            results[game_num] = -1
            player_two.black_wins += 1
            break
    black_ttable.clear()
    white_ttable.clear()
    del black_ttable
    del white_ttable
    del game


def find_best_player(player):
    return player.white_wins + player.black_wins


def genetic_algorithm(ngen=10, pop_number=10, gene_pool=HEURISTIC_WEIGHTS_RANGE, num_weights=len(heu.HEURISTIC_WEIGHTS), f_thresh=19, pmut=0.3):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    file_string = "train-"+now.strftime("%d_%m_%y-%H_%M")+".txt"
    file = open(f"train/{file_string}", "w", buffering=1)
    file.write(f'\n\tTRAINING - {dt_string}\n\n')
    print(f'\n\tTRAINING - {dt_string}\n\n')

    population = init_population(
        pop_number=pop_number,
        gene_pool=gene_pool,
        num_weights=num_weights
    )

    file.write("\tStarting Population\n")
    print("\tStarting Population\n")
    for player in population:
        file.write(str(player.weights)+"\n")
        print(player.weights)
    file.write("\n")

    for i in range(ngen):
        population = tournament(population)

        print(f'\n\tEnd tournament {i}\n')
        file.write(f'\n\tEnd tournament {i}\n')
        for player in population:
            file.write(str(player))
            print(player)

        if isinstance(population, TablutPlayer):
            print(f'\n\n\tBEST PLAYER\n')
            print(population)
            file.write('\n\n\tBEST PLAYER\n')
            file.write(str(population))
            break

        population = next_generation(
            population, find_best_player, gene_pool, f_thresh, pmut
        )

    if isinstance(population, list):
        print(f'\n\n\tLAST POPULATION\n')
        file.write('\n\n\tLAST POPULATION\n')
        for player in population:
            file.write(str(player))
            print(player)

    file.close()
    return population


def next_generation(population, fitness_fn, gene_pool, f_thresh, pmut):
    fittest_individual = fitness_threshold(fitness_fn, f_thresh, population)
    if fittest_individual:
        return fittest_individual
    population = [
        mutate(
            recombine_uniform(*weighted_select(2, population, fitness_fn)),
            gene_pool,
            pmut
        )
        for _ in range(len(population))
    ]
    return population


def fitness_threshold(fitness_fn, f_thresh, population):
    if not f_thresh:
        return None

    fittest_individual = max(population, key=fitness_fn)
    if fitness_fn(fittest_individual) >= f_thresh:
        return fittest_individual

    return None


def init_population(pop_number, gene_pool, num_weights):
    '''
    Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    num_weights:  Number of weights to set
    '''
    population = []
    random.seed(time.time())
    for _ in range(pop_number):
        new_weights = [
            utils.get_rand_double(gene_pool[0], gene_pool[-1])
            for _ in range(num_weights)
        ]
        population.append(TablutPlayer(weights=new_weights))
    return population


def weighted_select(r, population, fitness_fn):
    fitnesses = map(fitness_fn, population)
    return weighted_sample_with_replacement(r, population, fitnesses)


def weighted_sample_with_replacement(n, seq, weights):
    '''
    Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight.
    '''
    sampler = weighted_sampler(seq, weights)
    return [sampler() for _ in range(n)]


def weighted_sampler(seq, weights):
    '''
    Return a random-sample function that picks from seq weighted by weights.
    '''
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    index = bisect.bisect(totals, random.uniform(0, totals[-1]))
    return lambda: seq[
        index if index < len(seq) else utils.get_rand_int(0, len(seq))
    ]


def simple_select(r, population, fitness_fn):
    fitnesses = list(map(fitness_fn, population))
    results = list(zip(population, fitnesses))
    results.sort(key=lambda tup: tup[1])
    best_players = []
    for r in range(r):
        player, _ = results[r]
        best_players.append(player)
    return best_players


def recombine(x, y):
    c = random.randrange(0, len(x.weights))
    new_weights = x.weights[:c] + y.weights[c:]
    return TablutPlayer(weights=new_weights)


def recombine_uniform(x, y):
    n = len(x.weights)
    new_weights = [0] * n
    indexes = random.sample(range(n), n)
    for i, val in enumerate(indexes):
        new_weights[val] = x.weights[val] if i < n / 2 else y.weights[val]
    return TablutPlayer(weights=new_weights)


def mutate(x, gene_pool, pmut):
    if random.uniform(0, 1) >= pmut:
        return x

    c = random.randrange(0, len(x.weights))
    random.seed(time.time())
    new_gene = utils.get_rand_double(gene_pool[0], gene_pool[-1])
    new_weights = x.weights[:c] + [new_gene] + x.weights[c + 1:]
    return TablutPlayer(weights=new_weights)
