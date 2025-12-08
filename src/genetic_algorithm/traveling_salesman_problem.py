# 巡回セールスマン問題

import numpy as np
import random
import matplotlib.pyplot as plt

# --- 日本語フォント設定 ---
plt.rcParams['font.family']\
      = ['Meiryo', 'MS Gothic', 'Yu Gothic', 'DejaVu Sans']
plt.rcParams['font.sans-serif']\
      = ['Meiryo', 'MS Gothic', 'Yu Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# -------------------------

# --- 都市の定義 ---
NUM_CITIES = 20
np.random.seed(42)  # 結果の再現性のためにシードを設定
cities = np.random.rand(NUM_CITIES, 2) * 100  # 100x100の範囲で20個の都市をランダム生成

# --- GAパラメータ ---
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
MUTATION_RATE = 0.01


def calculate_distance(city1, city2):
    """二つの都市間のユークリッド距離を計算する"""
    return np.sqrt(np.sum((city1 - city2)**2))


def calculate_total_distance(route):
    """特定の経路（順序）の総距離（コスト）を計算する"""
    total_dist = 0
    # ルート内の隣接する都市間の距離を合計
    for i in range(len(route) - 1):
        city_index_1 = route[i]
        city_index_2 = route[i+1]
        total_dist +=\
            calculate_distance(cities[city_index_1], cities[city_index_2])

    # 最後の都市から最初の都市に戻る距離を追加（巡回）
    total_dist += calculate_distance(cities[route[-1]], cities[route[0]])
    return total_dist


# --- 1. 初期集団の生成 ---
def create_initial_population():
    population = []
    initial_route = list(range(NUM_CITIES))
    for _ in range(POPULATION_SIZE):
        # 都市の順序をランダムにシャッフルして経路を生成
        random.shuffle(initial_route)
        population.append(initial_route.copy())
    return population


# --- 2. 適応度の計算 ---
def get_fitness(route):
    """適応度は総距離の逆数とする (距離が短いほど良い)"""
    distance = calculate_total_distance(route)
    # 距離が0になることはないので、1/距離を適応度とする
    return 1.0 / distance


# --- 3. 選択 (トーナメント選択) ---
def selection(population, fitnesses):
    """ルーレット選択やトーナメント選択などが一般的。ここではトーナメント選択を採用。"""
    K = 5  # トーナメントサイズ
    tournament = random.choices(list(zip(population, fitnesses)), k=K)
    # 適応度が最も高い個体（経路）を選択
    return max(tournament, key=lambda x: x[1])[0]


# --- 4. 交叉 (順序交叉: Order Crossover, OX) ---
def crossover(parent1, parent2):
    """順序を維持する交叉 (TSPに適している)"""
    size = len(parent1)
    child = [-1] * size

    # 交叉点のランダム選択
    start, end = sorted(random.sample(range(size), 2))

    # 親1から中央の部分をそのままコピー
    child[start:end] = parent1[start:end]

    # 親2から残りの要素を順序を保ちながらコピー
    parent2_ptr = 0
    child_ptr = 0

    while child_ptr < size:
        if child[child_ptr] == -1:  # まだ埋まっていない場所
            # 親2の要素が子にまだ含まれていないか確認
            if parent2[parent2_ptr] not in child:
                child[child_ptr] = parent2[parent2_ptr]
                child_ptr += 1
            parent2_ptr += 1
        else:  # すでに埋まっている場所
            child_ptr += 1

    return child


# --- 5. 突然変異 (2点交換: Swap Mutation) ---
def mutate(route):
    """経路内の2つの都市をランダムに交換する"""
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


def genetic_algorithm():
    # 1. 初期集団の生成
    population = create_initial_population()

    # 初期集団の最良個体で best_route を直接初期化する
    fitnesses = [get_fitness(route) for route in population]
    current_best_index = np.argmax(fitnesses)

    # リスト型で初期化
    best_route = population[current_best_index].copy()
    min_distance = calculate_total_distance(best_route)
    history_distances = [min_distance]

    for generation in range(NUM_GENERATIONS):
        # 2. 適応度の計算
        fitnesses = [get_fitness(route) for route in population]

        # 最良個体の更新
        current_best_index = np.argmax(fitnesses)
        current_best_route = population[current_best_index]
        current_min_distance = calculate_total_distance(current_best_route)
        best_route = population[current_best_index].copy()

        if current_min_distance < min_distance:
            min_distance = current_min_distance
            best_route = current_best_route

        history_distances.append(min_distance)

        # 次世代の生成
        new_population = []

        # エリート戦略: 最良個体をそのまま次世代に残す (探索の収束を助ける)
        new_population.append(best_route.copy())

        while len(new_population) < POPULATION_SIZE:
            # 3. 選択
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)

            # 4. 交叉
            child = crossover(parent1, parent2)

            # 5. 突然変異
            child = mutate(child)

            new_population.append(child)

        population = new_population

        # ログ出力
        if (generation + 1) % 50 == 0:
            print(f"世代 {generation+1}: 最小距離 = {min_distance:.2f}")

    print("\n--- 探索終了 ---")
    print(f"最終的な最小距離: {min_distance:.2f}")
    return best_route, history_distances


# --- 実行 ---
best_route, history_distances = genetic_algorithm()


# --- 経路の描画 ---
def plot_route(route, cities, title):
    plt.figure(figsize=(8, 6))

    # 巡回経路の座標リストを生成
    ordered_cities = cities[route + [route[0]]]  # 最後に最初の都市を再度追加してループを閉じる

    # 都市の点
    plt.plot(cities[:, 0], cities[:, 1], 'o', color='blue', label='Cities')

    # 経路の線
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1],
             '-', color='red', label='Route')

    # 都市番号のラベル付け
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x + 1, y + 1))

    plt.title(
        f"{title}\nTotal Distance: {calculate_total_distance(route):.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


# --- 描画実行 ---
plot_route(best_route, cities, "GAによるTSPの最適経路")

# --- 収束グラフの描画 ---
plt.figure(figsize=(8, 4))
plt.plot(history_distances)
plt.title("世代ごとの最小距離の推移 (収束グラフ)")
plt.xlabel("世代 (Generation)")
plt.ylabel("最小距離 (Min Distance)")
plt.grid(True)
plt.show()
