import random

# 品物データ: (重量, 価値)
ITEMS = [(4, 12), (2, 2), (1, 1), (10, 4), (1, 4), (5, 15), (7, 10), (3, 5)]
N_ITEMS = len(ITEMS)  # 品物の数
MAX_CAPACITY = 15    # ナップサックの最大容量

# GAパラメータ
POPULATION_SIZE = 50   # 集団サイズ
N_GENERATIONS = 100    # 世代数
PROB_CROSSOVER = 0.8   # 交叉確率
PROB_MUTATION = 0.05   # 突然変異確率


def create_individual():
    """ランダムな個体（染色体）を生成する"""
    return [random.randint(0, 1) for _ in range(N_ITEMS)]


def create_initial_population(size):
    """初期集団を生成する"""
    return [create_individual() for _ in range(size)]


def evaluate_fitness(individual):
    """個体の適応度（総価値）を評価する"""
    current_weight = 0
    current_value = 0

    # 選択された品物の重量と価値を計算
    for i in range(N_ITEMS):
        if individual[i] == 1:
            weight, value = ITEMS[i]
            current_weight += weight
            current_value += value

    # 制約条件の処理
    if current_weight > MAX_CAPACITY:
        return 0  # 容量を超過した場合は適応度を0とする
    else:
        return current_value


def selection(population, fitnesses, k=3):
    """トーナメント選択により親を選ぶ"""
    # 集団からランダムにk個体を選び、その中で最も適応度が高い個体を親とする
    selected_parent = None
    best_fitness = -1

    for _ in range(k):
        idx = random.randint(0, len(population) - 1)
        if fitnesses[idx] > best_fitness:
            best_fitness = fitnesses[idx]
            selected_parent = population[idx]
    return selected_parent


def crossover(parent1, parent2):
    """一点交叉を実行する"""
    if random.random() < PROB_CROSSOVER:
        # 交叉点をランダムに選択
        crossover_point = random.randint(1, N_ITEMS - 1)

        # 遺伝子の交換
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2


def mutation(individual):
    """突然変異を実行する"""
    for i in range(N_ITEMS):
        if random.random() < PROB_MUTATION:
            # 遺伝子を反転させる (0 -> 1, 1 -> 0)
            individual[i] = 1 - individual[i]
    return individual


def run_ga():
    # 初期集団の生成
    population = create_initial_population(POPULATION_SIZE)

    # 最良解を追跡するための変数
    best_individual = None
    best_fitness_global = -1

    for gen in range(N_GENERATIONS):
        # 1. 適応度の評価
        fitnesses = [evaluate_fitness(ind) for ind in population]

        # 2. 最良個体の更新
        current_best_fitness = max(fitnesses)
        current_best_index = fitnesses.index(current_best_fitness)

        if current_best_fitness > best_fitness_global:
            best_fitness_global = current_best_fitness
            best_individual = population[current_best_index]

        print(f"世代 {gen}: 最良適応度 = {best_fitness_global}")

        # 3. 次世代の生成
        next_population = []
        while len(next_population) < POPULATION_SIZE:
            # 選択
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)

            # 交叉
            child1, child2 = crossover(parent1, parent2)

            # 突然変異
            next_population.append(mutation(child1))
            if len(next_population) < POPULATION_SIZE:
                next_population.append(mutation(child2))

        # 世代交代
        population = next_population

    print("\n--- 実行結果 ---")
    print(f"最良の総価値 (適応度): {best_fitness_global}")
    print(f"最良の組み合わせ: {best_individual}")


# 実行
if __name__ == "__main__":
    run_ga()
