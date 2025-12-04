import numpy as np
import matplotlib.pyplot as plt

# --- パラメータ設定 ---
MU = 10         # 親の数
LAMBDA = 50     # 子の数 (λ > μ である必要あり)
SIGMA = 5.0     # 突然変異の初期強度 (ステップサイズ)
GENE_LENGTH = 2  # (x, y) の2次元
MAX_GENERATIONS = 100
SEARCH_RANGE = 100.0  # 探索範囲 [-100, 100]


# --- 目的関数 (Schaffer's F6関数: 最小化) ---
def schaffers_f6(individual):
    """個体（[x, y]）の目的関数の値を計算する"""
    x, y = individual[0], individual[1]
    numerator = np.sin(np.sqrt(x**2 + y**2))**2 - 0.5
    denominator = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 + numerator / denominator


# --- 初期化 ---
def initialize_population(size, length, search_range):
    """指定範囲でランダムな実数値で集団を初期化する"""
    # -100 から 100 の範囲で初期化
    return np.random.uniform(-search_range, search_range, size=(size, length))


def mutate(individual, sigma):
    """個体にガウスノイズを加える突然変異"""
    # 各遺伝子（x, y）に N(0, σ^2) の乱数を加算
    mutation_vector = np.random.normal(0, sigma, size=len(individual))
    mutated_individual = individual + mutation_vector

    # 探索範囲 [-100, 100] にクリップ（丸め込む）
    mutated_individual\
        = np.clip(mutated_individual, -SEARCH_RANGE, SEARCH_RANGE)

    return mutated_individual


def run_evolution_strategy():
    # 記録用リスト
    best_value_history = []

    # 1. 初期化
    parents = initialize_population(MU, GENE_LENGTH, SEARCH_RANGE)
    current_sigma = SIGMA  # σを固定（単純なES）

    for generation in range(MAX_GENERATIONS):
        # 2. 子 (λ) の生成
        children = []
        # 親μ個から、λ個の子を生成する
        for _ in range(LAMBDA):
            # 親をランダムに選ぶ（μ個から1体）
            parent_idx = np.random.randint(MU)
            parent = parents[parent_idx]

            # 突然変異を実行
            child = mutate(parent, current_sigma)
            children.append(child)

        children = np.array(children)

        # 3. 評価
        children_fitness = np.array([schaffers_f6(ind) for ind in children])

        # 世代ごとの最小値を記録
        best_value = np.min(children_fitness)
        best_value_history.append(best_value)
        print(f"世代 {generation}: 最小値 = {best_value:.6f}")

        # 4. 選択: (μ, λ)戦略
        # 子の適応度 (最小化) が小さい順にインデックスを取得
        sorted_indices = np.argsort(children_fitness)

        # λ個の子の中から、上位μ個を次世代の親とする
        next_parent_indices = sorted_indices[:MU]
        parents = children[next_parent_indices]

        # 終了条件の確認（例として、値が十分小さくなったら）
        if best_value < 1e-4:
            print(f"最適解に近い値が世代 {generation} で見つかりました。")
            break

    # 結果の可視化
    plot_results(best_value_history)

    # 最終的な最良個体を返す
    final_fitness = np.array([schaffers_f6(ind) for ind in parents])
    best_individual_index = np.argmin(final_fitness)
    return parents[best_individual_index], np.min(final_fitness)


# --- 可視化関数 ---
def plot_results(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Best Fitness (Min) in Generation')
    plt.axhline(y=0.0, color='r', linestyle='--', label='Optimal Value (0.0)')
    plt.xlabel('Generation')
    plt.ylabel('Function Value (Minimization)')
    plt.title("Evolution Strategy (ES) Optimization for Schaffer's F6")
    plt.grid(True)
    plt.legend()
    plt.show()


# --- 実行 ---
final_best_individual, final_best_value = run_evolution_strategy()
print("\n--- 最終結果 ---")
print(f"最終世代の最良個体: {final_best_individual}")
print(f"最終的な最小値: {final_best_value:.6f}")
