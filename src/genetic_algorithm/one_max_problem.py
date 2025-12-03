import numpy as np
import matplotlib.pyplot as plt

# --- 1. 定数と基本関数の定義（再掲） ---
POPULATION_SIZE = 50      # 個体数
GENE_LENGTH = 10          # 遺伝子の長さ (OneMaxなのでMAX_FITNESSは10)
MAX_GENERATIONS = 100     # 最大世代数
MUTATION_RATE = 0.05      # 突然変異率 (5%)
CROSSOVER_PROBABILITY = 0.8  # 交叉を実行する確率
TOURNAMENT_SIZE = 3       # トーナメント選択のサイズ


# 初期化関数
def initialize_population(pop_size, gene_length):
    return np.random.randint(0, 2, size=(pop_size, gene_length))


# 適応度関数 (OneMax)
def fitness_function(individual):
    return np.sum(individual)


# トーナメント選択関数
def tournament_selection(population, fitness_scores, k):
    pop_size = len(population)
    candidate_indices = np.random.choice(pop_size, k, replace=False)
    best_index\
        = candidate_indices[np.argmax(fitness_scores[candidate_indices])]
    return population[best_index].copy()


# 一点交叉関数
def single_point_crossover(parent1, parent2):
    gene_length = len(parent1)
    crossover_point = np.random.randint(1, gene_length)
    child1 = np.concatenate(
        [parent1[:crossover_point], parent2[crossover_point:]]
        )
    child2 = np.concatenate(
        [parent2[:crossover_point], parent1[crossover_point:]]
        )
    return child1, child2


# ビット反転変異関数
def bit_flip_mutation(individual, mutation_rate):
    # np.random.rand() < mutation_rate と比較して変異させるビットを決定
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    # XOR (^) を使って変異させる (1^1=0, 0^1=1)
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual


# --- 2. メインGAループの実行 ---

def run_genetic_algorithm():
    # 記録用リスト
    best_fitness_history = []

    # 1. 初期化
    current_population = initialize_population(POPULATION_SIZE, GENE_LENGTH)

    for generation in range(MAX_GENERATIONS):
        # 2. 評価
        fitness_scores = np.array(
            [fitness_function(ind) for ind in current_population]
            )

        # 世代ごとの最高適応度を記録
        best_fitness = np.max(fitness_scores)
        best_fitness_history.append(best_fitness)
        print(f"世代 {generation}: 最高適応度 = {best_fitness}")

        # 終了条件の確認: 最適解が見つかった場合
        if best_fitness == GENE_LENGTH:
            print(f"最適解が世代 {generation} で見つかりました。")
            break

        # 次世代の集団を格納するリスト
        next_population = []

        # 次世代の個体数が POPULATION_SIZE になるまで繰り返す
        while len(next_population) < POPULATION_SIZE:
            # 3. 選択: 親を2体選ぶ
            parent1 = tournament_selection(
                current_population, fitness_scores, TOURNAMENT_SIZE
                )
            parent2 = tournament_selection(
                current_population, fitness_scores, TOURNAMENT_SIZE
                )

            # 4. 生殖: 交叉と突然変異
            if np.random.rand() < CROSSOVER_PROBABILITY:
                # 交叉を実行
                child1, child2 = single_point_crossover(parent1, parent2)
            else:
                # 交叉しない場合は親がそのまま子となる (クローン)
                child1, child2 = parent1.copy(), parent2.copy()

            # 突然変異を実行
            child1 = bit_flip_mutation(child1, MUTATION_RATE)
            child2 = bit_flip_mutation(child2, MUTATION_RATE)

            # 次世代に追加
            next_population.append(child1)
            # 個体数がオーバーしないようにチェック
            if len(next_population) < POPULATION_SIZE:
                next_population.append(child2)

        # 5. 世代交代: 現集団を次集団で置き換える
        current_population = np.array(next_population)

    # 最終結果の可視化
    plot_results(best_fitness_history)

    # 最終的な最良個体を返す
    final_fitness\
        = np.array([fitness_function(ind) for ind in current_population])
    best_individual_index = np.argmax(final_fitness)
    return current_population[best_individual_index]


# --- 3. 可視化関数 ---
def plot_results(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Best Fitness in Generation')
    plt.axhline(y=GENE_LENGTH, color='r',
                linestyle='--', label='Optimal Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Genetic Algorithm Optimization for OneMax Problem')
    plt.grid(True)
    plt.legend()
    plt.show()


# --- 実行 ---
final_best_individual = run_genetic_algorithm()
final_fitness_score = fitness_function(final_best_individual)
print("\n--- 最終結果 ---")
print(f"最終世代の最良個体: {final_best_individual}")
print(f"最終的な最高適応度: {final_fitness_score} / {GENE_LENGTH}")
