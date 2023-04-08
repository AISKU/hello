import numpy as np

"""적응도 함수는 최적해와 가까워 질수록 더 높은 값을 return 해야함
그러므로 각각의 도시들의 거리를 계산한 뒤에 역수 형태를 취해서 더 적을 수록 큰 값이 반환되게 해야함"""
def fitness(path, map):
    dist = 0
    for i in range(len(path) - 1):
        """두 도시간의 길이 없거나 같은 도시가 중복될 경우에는 매우 큰 값을 주어서 페널티를 부여함"""
        if map.distances.get((path[i], path[i + 1]), np.inf) == np.inf or path.count(path[i]) > 1 :
            dist += 1000000
        else:
            dist += map.distances.get((path[i], path[i + 1]))
    
    """마지막 도시에서 처음 도시로 돌아오는 경로까지 계산해야함"""
    if map.distances.get((path[-1], path[0]), np.inf) == np.inf or path.count(path[-1]) > 1 :
        dist += 1000000
    else :
        dist += map.distances.get((path[-1], path[0]))
        
    """역수 형태 취하기"""
    fitness = 1 / dist

    return fitness