import heapq
import networkx as nx
import random
import numpy as np
import time
from coptpy import *

# 参数
n = 100
epsilon = 0.2

def create_spp_model(graph, source):
    env = Envr()
    model = env.createModel("SPP model")
    x = {}
    for edge in graph.edges:
        x[edge] = model.addVar(vtype=COPT.BINARY, name=f'x_{edge[0]}_{edge[1]}')
    obj = LinExpr()
    for edge in graph.edges:
        obj.addTerms(x[edge], graph.edges[edge]['weight'])
    model.setObjective(obj, COPT.MINIMIZE)
    for node in graph.nodes:
        if node != source:
            lhs = LinExpr()
            for edge in graph.edges:
                if node in edge:
                    coeff = 1 if edge[1] == node else -1
                    lhs.addTerms(x[edge], coeff)
            model.addConstr(lhs == 0, name=f'flow_conservation_{node}')

    # Add constraint to ensure flow starts from the source
    lhs_source = LinExpr()
    for edge in graph.edges:
        if source in edge:
            lhs_source.addTerms(x[edge], 1)
    model.addConstr(lhs_source == 1, name='flow_starts_from_source')

    return model

def dijkstra(graph, start):
    heap = [(0, start)]
    visited = set()
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0

    while heap:
        current_distance, current_vertex = heapq.heappop(heap)

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        for neighbor, edge_data in graph[current_vertex].items():
            weight = edge_data['weight']
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))

    return distances

def is_connected(graph):
    # Manual connectivity check
    visited = set()
    stack = [random.choice(list(graph.nodes))]

    while stack:
        current_vertex = stack.pop()
        visited.add(current_vertex)

        for neighbor in graph.neighbors(current_vertex):
            if neighbor not in visited:
                stack.append(neighbor)

    return len(visited) == len(graph.nodes)

def check_all_edges(graph):
    # Check for negative-weight edges
    for edge in graph.edges:
        if graph.edges[edge]['weight'] < 0:
            return True
    return False

# 随机生成每条边的权值
def generate_er_graph(n, p):
    graph = nx.erdos_renyi_graph(n, p)
    for edge in graph.edges:
        graph.edges[edge]['weight'] = random.randint(1, 10)
    return graph

# 运行Dijkstra算法和线性规划算法
def run(n):
    k = 20
    dijkstra_time_set = 0.0
    lp_time_set = 0.0

    while k >= 0:
        p_connected = ((1 + epsilon) * np.log(n)) / n + 0.001
        graph_connected = generate_er_graph(n, p_connected)
        if is_connected(graph_connected):
            k -= is_connected(graph_connected)
        else:
            graph_connected = generate_er_graph(n, p_connected)

        # Check for negative-weight edges
        if check_all_edges(graph_connected):
            continue

        start_node = random.choice(list(graph_connected.nodes))
        source_node = random.choice(list(graph_connected.nodes))

        # Dijkstra算法
        starttime = time.time()
        dijkstra_distances_connected = dijkstra(graph_connected, start_node)
        endtime = time.time()
        dijkstra_time_set += endtime - starttime

        # Create linear programming model
        spp_model = create_spp_model(graph_connected, start_node)
        start_time = time.time()
        spp_model.solve()
        end_time = time.time()
        lp_time_set += end_time - start_time

    print("Dijkstra求出规模为%s个点的图的最短路径耗时:" % n)
    print(dijkstra_time_set / 20)
    print("线性规划求解规模为%s个点的图的最短路径耗时:" % n)
    print(lp_time_set / 20)
run(100)
#run(2000)
#run(10000)