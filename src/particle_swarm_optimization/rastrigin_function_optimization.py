import numpy as np
import matplotlib.pyplot as plt

# --- パラメータ設定 ---
NUM_PARTICLES = 30
GENE_LENGTH = 2         # (x, y) の2次元
W = 0.7                 # 慣性重み
C1 = 2.0                # 自己認知係数 (pbestの引力)
C2 = 2.0                # 社会認知係数 (gbestの引力)
MAX_ITERATIONS = 100
SEARCH_RANGE = 5.12


# --- 目的関数 (Rastrigin関数: 最小化) ---
def rastrigin_function(individual):
    """個体（[x, y]）のRastrigin関数の値を計算する"""
    x = individual
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# --- 粒子クラスの定義 ---
class Particle:
    def __init__(self, length, search_range):
        # 位置 (x, y) をランダムに初期化
        self.position = np.random.uniform(-search_range, search_range, length)
        # 速度 (v) をゼロまたはランダムに初期化
        self.velocity = np.zeros(length)
        # pbest (個人最良位置) は初期位置と同じ
        self.pbest_position = self.position.copy()
        # pbest_value (個人最良値) は初期位置での適応度
        self.pbest_value = rastrigin_function(self.position)


def run_pso():
    # 記録用リスト
    best_value_history = []

    # 1. 初期化と gbest の設定
    particles = [
        Particle(GENE_LENGTH, SEARCH_RANGE) for _ in range(NUM_PARTICLES)
        ]

    # gbest の初期化: 初期集団で最も良い値と位置を探す
    gbest_position = particles[0].pbest_position.copy()
    gbest_value = particles[0].pbest_value

    for particle in particles:
        if particle.pbest_value < gbest_value:
            gbest_value = particle.pbest_value
            gbest_position = particle.pbest_position.copy()

    print(f"初期 gbest 値: {gbest_value:.6f}")

    for iteration in range(MAX_ITERATIONS):
        # 2. 粒子の更新
        for particle in particles:
            # 速度更新の計算
            r1 = np.random.rand(GENE_LENGTH)
            r2 = np.random.rand(GENE_LENGTH)

            # (1) 慣性項
            inertia_term = W * particle.velocity
            # (2) 自己認知項 (pbestへの引力)
            cognitive_term\
                = C1 * r1 * (particle.pbest_position - particle.position)
            # (3) 社会認知項 (gbestへの引力)
            social_term = C2 * r2 * (gbest_position - particle.position)

            # 新しい速度
            particle.velocity = inertia_term + cognitive_term + social_term

            # 位置の更新
            particle.position += particle.velocity

            # 探索範囲 [-5.12, 5.12] にクリップ
            particle.position\
                = np.clip(particle.position, -SEARCH_RANGE, SEARCH_RANGE)

            # 3. pbest の更新
            current_value = rastrigin_function(particle.position)
            if current_value < particle.pbest_value:
                particle.pbest_value = current_value
                particle.pbest_position = particle.position.copy()

            # 4. gbest の更新
            if particle.pbest_value < gbest_value:
                gbest_value = particle.pbest_value
                gbest_position = particle.pbest_position.copy()

        # 記録と表示
        best_value_history.append(gbest_value)
        print(f"反復 {iteration}: gbest 値 = {gbest_value:.6f}")

        if gbest_value < 1e-4:
            print(f"最適解に近い値が反復 {iteration} で見つかりました。")
            break

    # 結果の可視化
    plot_results(best_value_history)

    # 最終的な gbest を返す
    return gbest_position, gbest_value


# --- 可視化関数 ---
def plot_results(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Global Best Value (Min)')
    plt.axhline(y=0.0, color='r', linestyle='--', label='Optimal Value (0.0)')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value (Minimization)')
    plt.title("Particle Swarm Optimization (PSO) for Rastrigin Function")
    plt.grid(True)
    plt.legend()
    plt.show()


# --- 実行 ---
final_gbest_position, final_gbest_value = run_pso()
print("\n--- 最終結果 ---")
print(f"最終 gbest 位置: {final_gbest_position}")
print(f"最終 gbest 値: {final_gbest_value:.6f}")
