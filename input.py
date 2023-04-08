import math

"""입력 예시"""

"""초기 개체군 형성
유전자 pool이나 도시 개수는 map클래스의 인스턴스로 부터 불러오면 된다.
밑에는 총 개체 100개를 생성한 예시"""
population = init_population(100, "도시이름".get_gene_pool(), len("도시 이름"))
print(population[:5])

"""정답을 찾아 반환하는 부분
개체군, 적응도 함수, map인스턴스, 유전자pool를 넣으면 된다.
f_thres는 해당 값보다 짧은 해가 나왔을 때 탐색을 그만두는 경계값이다.
다만, 적응도함수는 역수 형태로 값이 나오기 때문에 경계값도 역수 형태로 넣어야 한다."""
solution = genetic_algorithm(population, fitness, "도시이름", gene_pool="도시이름".get_gene_pool(), f_thres=(1/1400))
print(solution, round(1/fitness(solution, "도시이름")), sep='\n')

"""입력값에는 여러가지 요소가 있다.
개체의 개수, 세대의 개수, 돌연변이 확률들을 적절히 조절하여 최적해를 찾을 수 있다.
그 외에도 개체군 형성 -> 해 찾기 라는 과정을 몇 번 반복해서 그 중에서 가장 좋은 최적해를 찾을 수도 있다."""