# ナップサック問題を動的計画法で解く

# 品物データ: (重量, 価値)
ITEMS = [(4, 12), (2, 2), (1, 1), (10, 4), (1, 4), (5, 15), (7, 10), (3, 5)]
N_ITEMS = len(ITEMS)  # 品物の数
MAX_CAPACITY = 15    # ナップサックの最大容量

# dp[i][j] : i番目の品物までを対象にした時の最大重量がjの場合の最大価値
dp = [[0] * (MAX_CAPACITY+1) for _ in range(N_ITEMS+1)]
for i in range(1, N_ITEMS+1):
    for j in range(1, MAX_CAPACITY+1):
        dp[i][j] = dp[i-1][j]
        # i番目の品物の重さと価値
        W = ITEMS[i-1][0]
        V = ITEMS[i-1][1]
        if j - W >= 0:
            dp[i][j] = max(dp[i][j], dp[i-1][j-W] + V)

print("\n--- 実行結果 ---")
print(f"最良の総価値 (適応度): {dp[N_ITEMS][MAX_CAPACITY]}")

# 組み合わせ復元コードの追加

# 最終的な最大価値から逆追跡を開始
current_capacity = MAX_CAPACITY
selected_items_indices = []

# 品物の番号iを N_ITEMS から 1 まで逆順にチェック
# DPテーブルは1始まりで定義しているため、iは1から始まる品物のインデックス
for i in range(N_ITEMS, 0, -1):
    W_i = ITEMS[i-1][0]  # i番目の品物の重量 (ITEMS配列は0始まり)
    V_i = ITEMS[i-1][1]  # i番目の品物の価値

    # 1. 品物 i を選ばなかった場合の価値 (dp[i-1][current_capacity])
    # 2. 現在の最大価値 (dp[i][current_capacity])

    # 品物 i を選ばなかった場合と選んだ場合を比較する
    # 現在の最大価値が、品物 i を選ばなかった場合(dp[i-1][current_capacity])よりも大きい場合、
    # かつ、容量が足りる場合、品物 i は選ばれている。
    is_max_value_changed = dp[i][current_capacity] != dp[i-1][current_capacity]
    if is_max_value_changed and current_capacity >= W_i:
        # 品物 i を選んだ場合
        selected_items_indices.append(i - 1)  # 0始まりのインデックスを記録
        # 容量を減らして、次の品物へ
        current_capacity -= W_i

    # そうでない場合、品物 i は選ばれていない
    # (dp[i][current_capacity] == dp[i-1][current_capacity] の時)
    # そのまま次の品物へ (容量 current_capacity は変わらない)

# 記録したインデックスを逆順にすると、品物の元の順番に戻る
selected_items_indices.reverse()

# 選ばれた品物リストを生成
slected_pattern\
     = [1 if i in selected_items_indices else 0 for i in range(N_ITEMS)]
selected_items = [ITEMS[idx] for idx in selected_items_indices]

print("\n--- 組み合わせの復元結果 ---")
print(f"最良の組み合わせ: {slected_pattern}")
print(f"最良の組み合わせ (重量, 価値): {selected_items}")

# 確認のための情報
total_weight = sum(ITEMS[idx][0] for idx in selected_items_indices)
total_value = sum(ITEMS[idx][1] for idx in selected_items_indices)

print(f"合計重量: {total_weight}")
print(f"合計価値: {total_value}")
