from collections import defaultdict

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