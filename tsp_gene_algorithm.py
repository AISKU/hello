from collections import defaultdict
import numpy as np
import operator
import random
import bisect
import math

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

"""def fitness(path, map):
    total_distance = 0
    check_list = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0, 'k': 0, 
                  'l': 0, 'm': 0, 'n': 0, 'o': 0, 'p': 0, 'q': 0, 'r': 0, 's': 0, 't': 0, 'u': 0, 'v': 0, 
                  'w': 0, 'x': 0, 'y': 0, 'z': 0 } # 0 : 아직 방문하지 않은 도시, 1 : 방문한 도시
    check_list[path[0]] = 1
    for i in range(len(path)-1):
        # 현재 도시 위치에서 다음 도시의 방문 여부를 판단
        # 만약, 이동할 다음 위치가 0(방문안함)이라면, map의 distance를 가져와서 total_distance에 대입
        if check_list[path[i+1]] == 0:
            temp = map.distances.get((path[i], path[i+1]))
            total_distance += temp
            check_list[path[i+1]] = 1
        # 만약, 이동할 다음 위치가 방문한 곳이라면, total_distance에 무한 숫자를 대입
        else: 
            total_distance += 100000
    
    # 처음 도시로 돌아와야 하므로 마지막 도시에서 처음 도시로의 경로를 더해줌
    if check_list[path[0]] == 0:
        total_distance += map.distances.get((path[-1], path[0]))
    else:
        total_distance += 100000

    return 1/total_distance """

"""적응도 함수는 최적해와 가까워 질수록 더 높은 값을 return 해야함
그러므로 각각의 도시들의 거리를 계산한 뒤에 역수 형태를 취해서 더 적을 수록 큰 값이 반환되게 해야함"""
def fitness(path, map):
    dist = 0
    for i in range(len(path) - 1):
        """두 도시간의 길이 없거나 같은 도시가 중복될 경우에는 매우 큰 값을 주어서 페널티를 부여함"""
        if path.count(path[i]) > 1 :
            dist += 1000000
        else:
            dist += map.distances.get((path[i], path[i + 1]))
    
    """마지막 도시에서 처음 도시로 돌아오는 경로까지 계산해야함"""
    if path.count(path[-1]) > 1 :
        dist += 1000000
    else :
        dist += map.distances.get((path[-1], path[0]))
        
    """역수 형태 취하기"""
    fitness = 1 / dist

    return fitness

class Map:
    """2차원 지도를 표현하는 그래프.
    Map(links, locations) 형태로 생성.
    - links: [(v1, v2)...] 또는 {(v1, v2): distance...} dict 구조 가능
    - locations: {v1: (x, y)} 형식으로 각 노드의 위치(좌표) 설정 가능
    - directed=False이면, 양방향 링크 추가. 즉, (v1, v2) 링크에 대해 (v2, v1) 링크 추가"""

    def __init__(self, links, locations=None, directed=False):
        if not hasattr(links, 'items'):
            links = {link: 1 for link in links}  # 거리 기본값을 1로 설정
        if not directed:
            for (v1, v2) in list(links):
                links[v2, v1] = links[v1, v2]
        self.distances = links
        self.locations = locations or defaultdict(lambda: (0, 0))
    
    """도시들의 개수를 반환하는 함수
    사용시에는 len(map이름)과 같이 사용하면 되고
    초기 개체군을 형성할 때 사용함"""
    def __len__(self):
        return len(self.locations)
    
    """존재하는 도시들을 리스트형태로 담아서 반환함
    gene_pool를 받아야 할 때 사용함"""
    def get_gene_pool(self):
        return list(self.locations.keys())
    
city_distance = Map({ ('a', 'z'): 75, ('a', 't'): 118, ('a', 'o'): 121, ('a', 's'): 140, 
                 ('a', 'c'): 160, ('a', 'l'): 96, ('a', 'f'): 120, ('a', 'm'): 112, ('a', 'd'): 134,
                 ('a', 'r'): 132, ('a', 'p'): 156, ('a', 'b'): 169,
                 ('z', 'o'): 71, ('z', 'f'): 81, ('z', 's'): 90, ('z', 'c'): 165, ('z', 'l'): 120, 
                 ('z', 't'): 76, ('z', 'm'): 110, ('z', 'd'): 137, ('z', 'r'): 69, ('z', 'p'): 98, 
                 ('z', 'b'): 168, ('o', 't'): 103, ('o', 's'): 151, ('o', 'r'): 102, 
                 ('o', 'f'): 65, ('o', 'p'): 153, ('o', 'b'): 170, ('o', 'l'): 78, ('o', 'm'): 99, 
                 ('o', 'd'): 102, ('o', 'c'): 106,
                 ('t', 'c'): 130, ('t', 'd'): 96, ('t', 'm'): 72, ('t', 'l'): 111, ('t', 'f'): 137, 
                 ('t', 's'): 60, ('t', 'r'): 100, ('t', 'p'): 120, ('t', 'b'): 130,
                 ('l', 's'): 50, ('l', 'm'): 70, ('l', 'c'): 66, ('l', 'r'): 55, ('l', 'b'): 124, 
                 ('l', 'f'): 102, ('l', 'd'): 55, ('l', 'p'): 88,
                 ('s', 'f'): 99, ('s', 'm'): 78, ('s', 'd'): 105, ('s', 'c'): 67, ('s', 'r'): 54, 
                 ('s', 'p'): 65, ('s', 'b'): 97,
                 ('f', 'm'): 77, ('f', 'd'): 116, ('f', 'c'): 112, ('f', 'r'): 43, ('f', 'p'): 88, 
                 ('f', 'b'): 211, ('m', 'd'): 75, ('m', 'c'): 65, ('m', 'r'): 83, ('m', 'p'): 130, ('m', 'b'): 137,
                 ('d', 'c'): 120, ('d', 'r'): 11, ('d', 'p'): 160, ('d', 'b'): 170,
                 ('c', 'r'): 146, ('c', 'p'): 138, ('c', 'b'): 77,
                 ('r', 'p'): 97, ('r', 'b'): 132, ('p', 'b'): 101 },
                {'a': (0, 0), 'b': (0, 0), 'c': (0, 0), 'd': (0, 0), 'f': (0, 0),
                 'l': (0, 0), 'm': (0, 0), 'o': (0, 0), 'p': (0, 0), 'r': (0, 0),
                 's': (0, 0), 't': (0, 0), 'z': (0, 0) })

    
"""초기 개체군 형성
유전자 pool이나 도시 개수는 map클래스의 인스턴스로 부터 불러오면 된다.
밑에는 총 개체 100개를 생성한 예시"""
population = init_population(100, city_distance.get_gene_pool(), len(city_distance))
print(population[:5])

"""정답을 찾아 반환하는 부분
개체군, 적응도 함수, map인스턴스, 유전자pool를 넣으면 된다.
f_thres는 해당 값보다 짧은 해가 나왔을 때 탐색을 그만두는 경계값이다.
다만, 적응도함수는 역수 형태로 값이 나오기 때문에 경계값도 역수 형태로 넣어야 한다."""
solution = genetic_algorithm(population, fitness, city_distance, gene_pool=city_distance.get_gene_pool(), f_thres=(1/1400))
print(solution, round(1/fitness(solution, city_distance)), sep='\n')

"""입력값에는 여러가지 요소가 있다.
개체의 개수, 세대의 개수, 돌연변이 확률들을 적절히 조절하여 최적해를 찾을 수 있다.
그 외에도 개체군 형성 -> 해 찾기 라는 과정을 몇 번 반복해서 그 중에서 가장 좋은 최적해를 찾을 수도 있다."""