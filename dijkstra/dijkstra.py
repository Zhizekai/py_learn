# 构造矩阵
def generate_matrix():
    M = 1E100
    matrix = [[0, 12, M, M, M, 16, 14],
              [12, 0, 10, M, M, 7, M],
              [M, 10, 0, 3, 5, 6, M],
              [M, M, 3, 0, 4, M, M],
              [M, M, 5, 4, 0, 2, 8],
              [16, 7, 6, M, 2, 0, 9],
              [14, M, M, M, 8, 9, 0]]
    return matrix


def dijkstra(matrix, source):
    M = 1E100
    n = len(matrix)  # 矩阵的行数
    m = len(matrix[0])  # 矩阵的列数
    if source >= n or n != m:
        print('Error!')
        return

    found = [source]  # 已找到最短路径的结点
    cost = [M] * n  # source到已找到最短路径的节点的最短距离
    cost[source] = 0
    path = [[]] * n  # source到其他节点的最短路径0
    path[source] = [source]
    while len(found) < n:  # 当已找到最短路径的节点小于n时
        min_value = M + 1
        col = -1
        row = -1
        for f in found:  # 以已找到最短路径的节点所在行为搜索对象
            for i in [x for x in range(n) if x not in found]:  # 只搜索没找出最短路径的列
                if matrix[f][i] + cost[f] < min_value:  # 找出最小值
                    min_value = matrix[f][i] + cost[f]  # 在某行找到最小值要加上source到该行的最短路径
                    row = f  # 记录所在行列
                    col = i
        if col == -1 or row == -1:  # 若没找出最小值且节点还未找完，说明图中存在不连通的节点
            break
        found.append(col)  # 在found中添加已找到的节点
        cost[col] = min_value  # source到该节点的最短距离即为min_value
        path[col] = path[row][:]  # 复制source到已找到节点的上一节点的路径
        path[col].append(col)  # 再其后添加已找到节点即为sorcer到该节点的最短路径
    return found, cost, path


def main():
    matrix = generate_matrix()
    found, cost, path = dijkstra(matrix, 3)
    print('found:')
    print(found)
    print('cost:')
    print(cost)
    print('path:')
    for p in path:
        print(p)


if __name__ == '__main__':
    main()
