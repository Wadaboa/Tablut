'''
Tablut states evaluation functions weights computation
'''


import time
import bisect
from multiprocessing import Process, Array
from datetime import datetime
import logging

import tablut_player.utils as utils
import tablut_player.config as conf
import tablut_player.strategy as strat
import tablut_player.heuristic as heu
from tablut_player.game import TablutGame


HEURISTICS_WEIGHTS_RANGE = [0, 20]

if conf.TRAIN:
    LOGGER = logging.getLogger('GeneticLogger')
    LOGGER_FILE_HANDLER = logging.FileHandler(
        f'train/train-{datetime.now().strftime("%d_%m_%y-%H_%M")}.log'
    )
    LOGGER_STREAM_HANDLER = logging.StreamHandler()
    LOGGER_FORMATTER = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    )
    LOGGER_FILE_HANDLER.setFormatter(LOGGER_FORMATTER)
    LOGGER_STREAM_HANDLER.setFormatter(LOGGER_FORMATTER)
    LOGGER.addHandler(LOGGER_FILE_HANDLER)
    LOGGER.addHandler(LOGGER_STREAM_HANDLER)
    LOGGER.setLevel(logging.INFO)


class TablutPlayer():
    '''
    Object representing an evaluated TablutGame player
    '''

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


def tournament(population, sleep=3):
    '''
    Play every game between every couple of players in the given population
    '''
    # Play
    game_num = 0
    processes = []
    results = Array('i', [0] * (len(population) ** 2))
    for player_one in population:
        for player_two in population:
            proc = Process(
                target=play,
                args=(player_one, player_two, results, game_num)
            )
            proc.start()
            processes.append(proc)
            game_num += 1
    time.sleep(sleep)
    for proc in processes:
        proc.join()

    # Collect results
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
    '''
    Play a single game between the given players and store results
    '''
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
    '''
    Genetic algorithm fitness function
    '''
    return player.white_wins + player.black_wins


def log_population(population, header=''):
    '''
    Log the given population, with the given header string
    '''
    string = f'{header}\n'
    for player in population:
        string += f'{player.weights}\n'
    LOGGER.info(string)


def genetic_algorithm(ngen=10,
                      pop_number=10,
                      gene_pool=HEURISTICS_WEIGHTS_RANGE,
                      num_weights=len(heu.HEURISTICS),
                      f_thresh=19,
                      pmut=0.3):
    '''
    Perform the given number of tournaments, with evolving populations
    '''
    population = init_population(
        pop_number=pop_number,
        gene_pool=gene_pool,
        num_weights=num_weights
    )
    log_population(population, header='Initial population')

    for i in range(ngen):
        population = tournament(population)
        log_population(population, header=f'End tournament {i}')

        if isinstance(population, TablutPlayer):
            log_population([population], header=f'Best player')
            break

        population = next_generation(
            population, find_best_player, gene_pool, f_thresh, pmut
        )

    if isinstance(population, list):
        log_population(population, header=f'Last population')

    return population


def next_generation(population, fitness_fn, gene_pool, f_thresh, pmut):
    '''
    Return the next generation of the given population
    '''
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
    '''
    Return the player that scored more than the given threshold, if it exists
    '''
    if not f_thresh:
        return None
    fittest_individual = max(population, key=fitness_fn)
    if fitness_fn(fittest_individual) >= f_thresh:
        return fittest_individual
    return None


def init_population(pop_number, gene_pool, num_weights):
    '''
    Initializes population for genetic algorithm
    '''
    population = []
    utils.set_random_seed()
    for _ in range(pop_number):
        new_weights = [
            utils.get_rand_double(gene_pool[0], gene_pool[-1])
            for _ in range(num_weights)
        ]
        population.append(TablutPlayer(weights=new_weights))
    return population


def weighted_select(num, population, fitness_fn):
    '''
    Select the best num players from the given population,
    weighted on the given fitness function
    '''
    fitnesses = map(fitness_fn, population)
    return weighted_sample_with_replacement(num, population, fitnesses)


def weighted_sample_with_replacement(num, seq, weights):
    '''
    Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight
    '''
    sampler = weighted_sampler(seq, weights)
    return [sampler() for _ in range(num)]


def weighted_sampler(seq, weights):
    '''
    Return a random sampler function that picks from seq weighted by weights
    '''
    totals = []
    for weight in weights:
        totals.append(weight + totals[-1] if totals else weight)
    index = bisect.bisect(totals, utils.get_rand_double(0, totals[-1]))
    return lambda: seq[
        index if index < len(seq) else utils.get_rand_int(0, len(seq))
    ]


def simple_select(num, population, fitness_fn):
    '''
    Select the best r players from the given population
    '''
    fitnesses = list(map(fitness_fn, population))
    results = list(zip(population, fitnesses))
    results.sort(key=lambda tup: tup[1])
    return [results[i] for i in range(num)]


def recombine(x, y):
    '''
    Crossover player x with player y and create a new player
    '''
    val = utils.get_rand_int(0, len(x.weights))
    new_weights = x.weights[:val] + y.weights[val:]
    return TablutPlayer(weights=new_weights)


def recombine_uniform(x, y):
    '''
    Recombine player x with player y and create a new player,
    by crossing over random subsets of weights
    '''
    num = len(x.weights)
    new_weights = [0] * num
    indexes = utils.get_rand(range(num), num)
    for i, val in enumerate(indexes):
        new_weights[val] = x.weights[val] if i < num / 2 else y.weights[val]
    return TablutPlayer(weights=new_weights)


def mutate(x, gene_pool, pmut):
    '''
    Perform a random mutation of player x, with the given probability
    '''
    if not utils.probability(pmut):
        return x

    val = utils.get_rand_int(0, len(x.weights))
    utils.set_random_seed()
    new_gene = utils.get_rand_double(gene_pool[0], gene_pool[-1])
    new_weights = x.weights[:val] + [new_gene] + x.weights[val + 1:]
    return TablutPlayer(weights=new_weights)
