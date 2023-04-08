import operator
import random
import bisect

def genetic_algorithm(population, fitness_fn, tspmap, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1, ):
    """유전 알고리즘.
        population: 초기 개체군
        fitness_fn: 적응도 함수
        gene_pool: 개체의 유전자들이 가질 수 있는 값들의 리스트
        f_thres: 적응도 임계치. 개체의 적응도가 이 값 이상이 되면 iteration이 멈춤.
        ngen: iteration 수
        pmut: 돌연변이 확률"""
    
    """ngen번 반복하여 다음 세대를 생성한다.
       기본적으로 유전 알고리즘은 선택(select) -> 교차(recombine) -> 변이(mutate)
       순으로 이루어진다. 각각의 함수에 대해서는 후에 나온다."""
    for i in range(ngen):
        population = [mutate(recombine(*select(2, population, fitness_fn, tspmap)), gene_pool, pmut)
                      for i in range(len(population))]
        
        fittest_individual = fitness_threshold(fitness_fn, f_thres, population, tspmap)
        if fittest_individual:
            return fittest_individual

    return max(population, key=lambda x: fitness_fn(x, tspmap))


def fitness_threshold(fitness_fn, f_thres, population, tspmap):
    """적응도 함수가 가장 높은 개체 리턴. 그 개체의 적응도가 임계치 미만이면 None 리턴"""
    if not f_thres:
        return None

    fittest_individual = max(population, key=lambda x: fitness_fn(x, tspmap))
    if fitness_fn(fittest_individual, tspmap) >= f_thres:
        return fittest_individual

    return None

def init_population(pop_number, gene_pool, state_length):
    """개체군 초기화.
    pop_number: 개체군에 포함될 개체 수
    gene_pool: 개체의 유전자들이 가질 수 있는 값들의 리스트
    state_length: 각 개체의 길이"""
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
        population.append(new_individual)

    return population

def select(r, population, fitness_fn, tspmap):
    """선택 연산(selection).
    r: 선택할 개체 수
    population: 개체군
    fitness_fn: 적응도 함수"""

    """적응도 함수를 사용할 때 map에 대한 정보도 필요하기 때문에
    lambda함수의 인자에 map에 대한 정보도 넘김"""
    fitnesses = map(lambda x: fitness_fn(x, tspmap), population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(r)]

def recombine(x, y):
    """교차 연산(point crossover): 하나의 교차점을 기준으로 결합됨"""
    n = len(x)
    c = random.randrange(0, n)
    return x[:c] + y[c:]

def mutate(x, gene_pool, pmut):
    """돌연변이 연산(mutation)"""
    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    new_gene = gene_pool[r]
    return x[:c] + [new_gene] + x[c + 1:]

def weighted_sampler(seq, weights):
    """weights의 가중치를 사용하여 seq를 랜덤 샘플링하는 함수 리턴."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]