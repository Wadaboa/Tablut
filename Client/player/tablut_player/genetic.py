'''
Tablut states evaluation functions weights computation
'''


import bisect
import logging
from multiprocessing import Pool
from datetime import datetime

import tablut_player.config as conf
import tablut_player.utils as utils
import tablut_player.strategy as strat
import tablut_player.heuristic as heu
from tablut_player.game import TablutGame


HEURISTICS_WEIGHTS_RANGE = [0, 20]
MAX_TURNS = 20


def init_logger():
    '''
    Initialize logger
    '''
    logger = logging.getLogger('GeneticLogger')
    logger_file_handler = logging.FileHandler(
        f'train/train-{datetime.now().strftime("%d_%m_%y-%H_%M")}.log'
    )
    logger_stream_handler = logging.StreamHandler()
    logger_formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    )
    logger_file_handler.setFormatter(logger_formatter)
    logger_stream_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_file_handler)
    logger.addHandler(logger_stream_handler)
    logger.setLevel(logging.INFO)
    return logger


class TablutPlayer():
    '''
    Object representing an evaluated TablutGame player
    '''

    def __init__(self, weights,
                 white_wins=0, black_wins=0, draws=0, num_games=-1):
        self.weights = weights
        self.white_wins = white_wins
        self.black_wins = black_wins
        self.draws = draws
        self.num_games = num_games

    def __eq__(self, other):
        for weight_one, weight_two in zip(self.weights, other.weights):
            if weight_one != weight_two:
                return False
        return True

    def __hash__(self):
        return hash(tuple(self.weights))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (
            f'W: {self.weights}'
            f' WW: {self.white_wins}'
            f' BW: {self.black_wins}'
            f' D: {self.draws}'
            f' NG: {self.num_games}\n'
        )

    def __repr__(self):
        return (
            f'Weights: {self.weights}\n'
            f'White wins: {self.white_wins}\n'
            f'Black wins: {self.black_wins}\n'
            f'Draws: {self.draws}\n'
            f'Number of games: {self.num_games}\n'
        )


def tournament(population):
    '''
    Play every game between every couple of players in the given population
    '''
    # Play
    args = []
    random_player = TablutPlayer(weights=[0] * len(heu.HEURISTICS))
    results = []
    population.add(random_player)
    for player_one in population:
        for player_two in population:
            if player_one != random_player or player_two != random_player:
                if all([weight == 0 for weight in player_one.weights]):
                    player_one_type = strat.random_player
                else:
                    player_one_type = conf.WHITE_PLAYER
                if all([weight == 0 for weight in player_two.weights]):
                    player_two_type = strat.random_player
                else:
                    player_two_type = conf.BLACK_PLAYER
                args.append((player_one, player_one_type,
                             player_two, player_two_type))

    with Pool() as proc:
        try:
            results = proc.starmap(play, args)
        except Exception as exp:
            print(exp)
            proc.terminate()
            proc.join()

    population.remove(random_player)

    # Collect results
    for game_num, res in enumerate(results):
        player_one, _, player_two, _ = args[game_num]
        if res == 1:
            player_one.white_wins += 1
        elif res == -1:
            player_two.black_wins += 1
        elif res == 0:
            player_one.draws += 1
            player_two.draws += 1
        player_one.num_games += 1
        player_two.num_games += 1
    return population


def play(player_one, player_one_type, player_two, player_two_type):
    '''
    Play a single game between the given players and store results
    '''
    game = TablutGame()
    game_state = game.initial
    black_ttable = strat.TT()
    white_ttable = strat.TT()
    winner = 0
    while not game.terminal_test(game_state) and game.turn < MAX_TURNS:
        if game.turn % 10 == 0:
            black_ttable.clear()
            white_ttable.clear()
        game.inc_turn()

        heu.set_heuristic_weights(player_one.weights)

        white_move = strat.get_move(
            game, game_state, player_one_type, prev_move=None,
            timeout=conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD,
            max_depth=0, tt=white_ttable
        )
        game_state = game.result(game_state, white_move)
        if game.terminal_test(game_state):
            winner = 1
            break

        heu.set_heuristic_weights(player_two.weights)

        black_move = strat.get_move(
            game, game_state, player_two_type, prev_move=None,
            timeout=conf.MOVE_TIMEOUT - conf.MOVE_TIME_OVERHEAD,
            max_depth=0, tt=black_ttable
        )
        game_state = game.result(game_state, black_move)
        if game.terminal_test(game_state):
            winner = -1
            break
    if game.turn >= MAX_TURNS:
        winner = MAX_TURNS
    black_ttable.clear()
    white_ttable.clear()
    return winner


def find_best_player(player):
    '''
    Genetic algorithm fitness function
    '''
    return player.white_wins + player.black_wins


def log_population(logger, population, header=''):
    '''
    Log the given population, with the given header string
    '''
    string = f'{header}\n'
    for player in population:
        string += f'{player}\n'
    logger.info(string)


def genetic_algorithm(ngen=10,
                      pop_number=10,
                      gene_pool=HEURISTICS_WEIGHTS_RANGE,
                      num_weights=len(heu.HEURISTICS),
                      pmut=0.4):
    '''
    Perform the given number of tournaments, with evolving populations
    '''
    utils.set_random_seed()
    logger = init_logger()
    f_thresh = int(0.90 * ((pop_number + 1) ** 2) - 1)
    population = init_population(
        pop_number=pop_number,
        gene_pool=gene_pool,
        num_weights=num_weights
    )
    log_population(logger, population, header='Initial population')

    for i in range(ngen):
        population = tournament(population)
        log_population(logger, population, header=f'End tournament {i}')
        if i < ngen - 1:
            population = next_generation(
                population, find_best_player, gene_pool, f_thresh, pmut
            )
            if isinstance(population, TablutPlayer):
                log_population(logger, [population], header=f'Best player')
                break

    if isinstance(population, list):
        log_population(logger, population, header=f'Last population')

    return population


def next_generation(population, fitness_fn, gene_pool, f_thresh, pmut):
    '''
    Return the next generation of the given population
    '''
    fittest_individual = fitness_threshold(fitness_fn, f_thresh, population)
    if fittest_individual:
        return fittest_individual
    new_population = set()
    while len(new_population) < len(population):
        new_population.add(
            mutate(
                recombine_uniform(*weighted_select(2, population, fitness_fn)),
                gene_pool,
                pmut
            )
        )

    return new_population


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
    population = set()
    population.add(TablutPlayer(weights=list(heu.HEURISTICS.values())))
    while len(population) < pop_number:
        new_weights = [
            utils.get_rand_double(gene_pool[0], gene_pool[-1])
            for _ in range(num_weights)
        ]
        population.add(TablutPlayer(weights=new_weights))
    return population


def weighted_select(num, population, fitness_fn):
    '''
    Select the best num players from the given population,
    weighted on the given fitness function
    '''
    pop = list(population)
    fitnesses = map(fitness_fn, pop)
    return weighted_sample_with_replacement(num, pop, fitnesses)


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
    new_gene = utils.get_rand_double(gene_pool[0], gene_pool[-1])
    new_weights = x.weights[:val] + [new_gene] + x.weights[val + 1:]
    return TablutPlayer(weights=new_weights)
