import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree


class NSGADATIS:

    def __init__(self, test_features, uncertainty_scores, budget=500,
                 pop_size=100, generations=100, crossover_rate=0.9,
                 mutation_rate=0.01, pca_components=100):
        """
            test_features: 测试集的潜在特征 [n_samples, feature_dim]
            uncertainty_scores: 测试输入的不确定性分数 [n_samples]
            budget: 需选择的测试输入数量
            pop_size: 种群大小
            generations: 迭代次数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            pca_components: 用于几何多样性计算的PCA降维维度
        """
        self.test_features = Normalizer(norm='l2').fit_transform(test_features)
        self.uncertainty_scores = uncertainty_scores
        self.budget = budget
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_samples = test_features.shape[0]
        self.original_dim = test_features.shape[1]

        # PCA降维处理
        self.use_pca = (self.original_dim > pca_components)
        if self.use_pca:
            print(f"特征维度较高 ({self.original_dim})，使用PCA降维至{pca_components}维")
            self.pca = PCA(n_components=pca_components)
            self.pca.fit(self.test_features)
            self.reduced_features = self.pca.transform(self.test_features)
            print(f"PCA解释方差比例: {sum(self.pca.explained_variance_ratio_):.4f}")
        else:
            self.reduced_features = self.test_features

        # 为几何多样性准备的Min-Max缩放器
        self.scaler = MinMaxScaler()
        self.scaled_features = self.scaler.fit_transform(self.reduced_features)

    def _initialize_population(self):
        # 确保种群中的每个个体都是唯一索引的集合
        population = []

        # 添加不确定性最大个体
        uncertainty_sorted = np.argsort(-self.uncertainty_scores)
        greedy_by_uncertainty = uncertainty_sorted[:self.budget].tolist()
        population.append(greedy_by_uncertainty)

        for _ in range(self.pop_size - 1):
            individual = random.sample(range(self.n_samples), self.budget)
            population.append(individual)
        return population

    def _calculate_geometric_diversity(self, individual):
        features = self.scaled_features[individual]

        dot_p = np.dot(features, features.T)
        _, diversity = np.linalg.slogdet(dot_p)

        return diversity

    def _calculate_objectives(self, individual):
        uncertainty = np.mean(self.uncertainty_scores[individual])

        diversity = self._calculate_geometric_diversity(individual)

        coverage = self._estimate_fault_coverage(individual)

        # 返回三个目标
        return [uncertainty, diversity, coverage]

    def _evaluate_fault_detection(self, pareto_front, pareto_objectives):
        scores = []
        alpha = 100
        for i, solution in enumerate(pareto_front):
            # 结合原始DATIS的评分方法
            certainty_score = pareto_objectives[i][0]  # 不确定性
            diversity_score = pareto_objectives[i][1]  # 多样性

            if len(pareto_objectives[i]) > 2:
                coverage_score = pareto_objectives[i][2]  # 已计算的故障覆盖
            else:
                coverage_score = self._estimate_fault_coverage(solution)

            # 加权组合评分
            combined_score = (certainty_score * alpha / self.n_samples +
                              diversity_score / (self.budget ^ 2) +
                              coverage_score)
            scores.append(combined_score)

        best_idx = np.argmax(scores)
        return best_idx

    def _fast_non_dominated_sort(self, population):
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        # 计算每个解的支配关系
        objectives = [self._calculate_objectives(ind) for ind in population]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # 检查i是否支配j
                if all(o_i >= o_j for o_i, o_j in zip(objectives[i], objectives[j])) and \
                        any(o_i > o_j for o_i, o_j in zip(objectives[i], objectives[j])):
                    dominated_solutions[i].append(j)
                # 检查j是否支配i
                elif all(o_j >= o_i for o_i, o_j in zip(objectives[i], objectives[j])) and \
                        any(o_j > o_i for o_i, o_j in zip(objectives[i], objectives[j])):
                    domination_count[i] += 1

            # 如果i不被任何解支配，将其添加到第一前沿
            if domination_count[i] == 0:
                fronts[0].append(i)

        # 生成所有前沿
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for j in fronts[i]:
                for k in dominated_solutions[j]:
                    domination_count[k] -= 1
                    if domination_count[k] == 0:
                        next_front.append(k)
            i += 1
            if next_front:  # 只有当next_front非空时才添加
                fronts.append(next_front)

        return fronts, objectives

    def _calculate_crowding_distance(self, front, objectives):
        """计算拥挤度距离"""
        if len(front) <= 2:
            return {i: float('inf') for i in front}

        distance = {i: 0 for i in front}
        front_size = len(front)
        obj_dim = len(objectives[0])

        for m in range(obj_dim):
            # 按第m个目标函数值排序
            sorted_front = sorted(front, key=lambda i: objectives[i][m])

            # 边界点设为无穷大
            distance[sorted_front[0]] = float('inf')
            distance[sorted_front[-1]] = float('inf')

            # 计算中间点的拥挤度
            f_max = objectives[sorted_front[-1]][m]
            f_min = objectives[sorted_front[0]][m]

            # 避免除零错误
            if f_max == f_min:
                continue

            for i in range(1, front_size - 1):
                distance[sorted_front[i]] += (objectives[sorted_front[i + 1]][m] -
                                              objectives[sorted_front[i - 1]][m]) / (f_max - f_min)

        return distance

    def _crowded_comparison_operator(self, i, j, rank, distance):
        if rank[i] < rank[j]:
            return True
        elif rank[i] > rank[j]:
            return False
        elif distance[i] > distance[j]:
            return True
        return False


    def _tournament_selection_single(self, population_indices, rank, distance, k=2):
        candidates = random.sample(population_indices, k)
        best = candidates[0]
        for candidate in candidates[1:]:
            if self._crowded_comparison_operator(candidate, best, rank, distance):
                best = candidate
        return best

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            # 计算父代之间的差异
            set1 = set(parent1)
            set2 = set(parent2)

            # 共有元素和差异元素
            common = list(set1.intersection(set2))
            diff1 = list(set1 - set2)
            diff2 = list(set2 - set1)

            # 优先保留共有元素，然后从差异元素中随机选择
            child = common.copy()

            # 从差异中选择元素
            remaining = self.budget - len(child)
            combined_diff = diff1 + diff2

            if combined_diff and remaining > 0:
                # 优先选择不确定性较高的元素
                diff_scores = [self.uncertainty_scores[idx] for idx in combined_diff]
                sorted_indices = np.argsort(-np.array(diff_scores))

                deterministic_count = int(remaining * 0.7)
                selected_deterministic = [combined_diff[sorted_indices[i]]
                                          for i in range(min(deterministic_count, len(sorted_indices)))]

                remaining_after_deterministic = remaining - len(selected_deterministic)
                if remaining_after_deterministic > 0 and len(combined_diff) > len(selected_deterministic):
                    remaining_pool = [idx for i, idx in enumerate(combined_diff)
                                      if i not in sorted_indices[:len(selected_deterministic)]]
                    selected_random = random.sample(remaining_pool,
                                                    min(remaining_after_deterministic, len(remaining_pool)))
                else:
                    selected_random = []

                child.extend(selected_deterministic + selected_random)

            # 如果仍未达到预算，从未使用的样本中随机选择
            if len(child) < self.budget:
                unused = list(set(range(self.n_samples)) - set(child))
                if unused:
                    additional = random.sample(unused, min(self.budget - len(child), len(unused)))
                    child.extend(additional)

            # 如果超出预算，随机移除
            if len(child) > self.budget:
                child = random.sample(child, self.budget)

            return child
        return parent1.copy()

    def _estimate_fault_coverage(self, individual):
        # 根据特征相似性将测试集划分为潜在的故障类型簇
        if not hasattr(self, 'feature_clusters'):
            # 预计算特征聚类（模拟不同故障类型），
            from sklearn.cluster import KMeans
            n_clusters = min(20, self.n_samples // 100)  # 估计故障类型数量
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.feature_clusters = self.kmeans.fit_predict(self.reduced_features)

        # 计算所选样本覆盖的簇数量
        selected_clusters = set(self.feature_clusters[individual])
        coverage_ratio = len(selected_clusters) / len(np.unique(self.feature_clusters))

        return coverage_ratio

    def _mutation(self, individual):
        individual = individual.copy()
        individual_set = set(individual)

        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                # 移除当前元素
                old_value = individual[i]
                individual_set.remove(old_value)

                # 生成一个不在individual_set中的随机索引
                tries = 0
                while tries < 10:
                    new_index = random.randint(0, self.n_samples - 1)
                    if new_index not in individual_set:
                        individual[i] = new_index
                        individual_set.add(new_index)
                        break
                    tries += 1

                # 如果多次尝试后仍找不到新索引，保留原值
                if tries >= 10:
                    individual[i] = old_value
                    individual_set.add(old_value)

        return individual

    def run(self):
        population = self._initialize_population()

        # 主循环
        with tqdm(total=self.generations, desc="NSGA-II Optimization") as pbar:
            for generation in range(self.generations):
                # 计算当前种群的Pareto前沿和拥挤度
                try:
                    fronts, objectives = self._fast_non_dominated_sort(population)

                    # 计算每个前沿的拥挤度
                    crowding_distances = {}
                    for front in fronts:
                        distances = self._calculate_crowding_distance(front, objectives)
                        crowding_distances.update(distances)

                    # 确定每个个体的排名
                    ranks = {}
                    for i, front in enumerate(fronts):
                        for j in front:
                            ranks[j] = i

                    # 生成子代
                    offspring = []
                    population_indices = list(range(len(population)))

                    for _ in range(self.pop_size // 2):
                        # 使用锦标赛选择两个父代
                        parent1_idx = self._tournament_selection_single(
                            population_indices, ranks, crowding_distances, k=2)
                        parent2_idx = self._tournament_selection_single(
                            population_indices, ranks, crowding_distances, k=2)

                        parent1 = population[parent1_idx]
                        parent2 = population[parent2_idx]

                        # 交叉
                        child1 = self._crossover(parent1, parent2)
                        child2 = self._crossover(parent2, parent1)

                        # 变异
                        child1 = self._mutation(child1)
                        child2 = self._mutation(child2)

                        offspring.extend([child1, child2])

                    # 合并父代和子代
                    combined = population + offspring

                    combined_fronts, combined_objectives = self._fast_non_dominated_sort(combined)

                    combined_crowding_distances = {}
                    for front in combined_fronts:
                        distances = self._calculate_crowding_distance(front, combined_objectives)
                        combined_crowding_distances.update(distances)

                    combined_ranks = {}
                    for i, front in enumerate(combined_fronts):
                        for j in front:
                            combined_ranks[j] = i

                    new_population = []
                    front_idx = 0
                    while (len(new_population) + len(combined_fronts[front_idx]) <= self.pop_size
                           and front_idx < len(combined_fronts)):
                        # 完全添加当前前沿
                        for i in combined_fronts[front_idx]:
                            new_population.append(combined[i])
                        front_idx += 1

                    if len(new_population) < self.pop_size and front_idx < len(combined_fronts):
                        last_front = combined_fronts[front_idx]
                        # 按拥挤度排序
                        sorted_last_front = sorted(last_front,
                                                   key=lambda i: combined_crowding_distances.get(i, 0),
                                                   reverse=True)

                        remaining = self.pop_size - len(new_population)
                        for i in range(min(remaining, len(sorted_last_front))):
                            new_population.append(combined[sorted_last_front[i]])

                    while len(new_population) < self.pop_size:
                        random_idx = random.randint(0, len(combined) - 1)
                        new_population.append(combined[random_idx])

                    # 更新种群
                    population = [ind.copy() for ind in new_population]

                except Exception as e:
                    print(f"第{generation + 1}代迭代中发生错误: {str(e)}")

                pbar.update(1)

        try:
            # 返回最后一代的非支配解
            final_fronts, final_objectives = self._fast_non_dominated_sort(population)

            if not final_fronts or not final_fronts[0]:
                print("警告：未找到任何帕累托最优解，返回随机解")
                return random.choice(population), [random.choice(population)], [[0, 0, 0]]

            pareto_front = [population[i] for i in final_fronts[0]]
            pareto_objectives = [final_objectives[i] for i in final_fronts[0]]

            # 使用优化的解选择方法
            best_idx = self._evaluate_fault_detection(pareto_front, pareto_objectives)

            # 打印帕累托前沿信息
            print(f"发现{len(pareto_front)}个帕累托最优解")
            if pareto_objectives:
                print(
                    f"不确定性范围: [{min(obj[0] for obj in pareto_objectives):.4f}, {max(obj[0] for obj in pareto_objectives):.4f}]")
                print(
                    f"多样性范围: [{min(obj[1] for obj in pareto_objectives):.4f}, {max(obj[1] for obj in pareto_objectives):.4f}]")

                # 如果有第三个目标（故障覆盖率）
                if len(pareto_objectives[0]) > 2:
                    print(
                        f"故障覆盖率范围: [{min(obj[2] for obj in pareto_objectives):.4f}, {max(obj[2] for obj in pareto_objectives):.4f}]")

                print(f"选择的解 - 不确定性: {pareto_objectives[best_idx][0]:.4f}, 多样性: {pareto_objectives[best_idx][1]:.4f}",
                      end="")

                # 如果有第三个目标
                if len(pareto_objectives[best_idx]) > 2:
                    print(f", 故障覆盖率: {pareto_objectives[best_idx][2]:.4f}")
                else:
                    print("")

            return pareto_front[best_idx], pareto_front, pareto_objectives

        except Exception as e:
            print(f"计算最终结果时出错: {str(e)}")
            # 在出错时返回随机解
            return random.choice(population), [random.choice(population)], [[0, 0, 0]]  # 更新为三个目标


if __name__ == "__main__":
    n_samples = 10000
    feature_dim = 256
    test_features = np.random.rand(n_samples, feature_dim)  # 潜在特征
    uncertainty_scores = np.random.rand(n_samples)  # 不确定性分数

    # 初始化并运行NSGA-II
    nsga2 = NSGADATIS(
        test_features=test_features,
        uncertainty_scores=uncertainty_scores,
        budget=500,  # 选择500个测试输入
        pop_size=100,  # 种群大小
        generations=100,  # 迭代次数
        pca_components=50  # PCA降维至50维
    )

    # 返回最佳解和帕累托前沿
    best_solution, pareto_front, pareto_objectives = nsga2.run()
    print(f"选择的测试输入数量: {len(best_solution)}")