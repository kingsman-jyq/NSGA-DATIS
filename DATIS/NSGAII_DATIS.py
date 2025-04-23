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

        # 方法1：计算几何多样性 GD = det(F·F^T)
        # 获取已缩放的特征矩阵

        # dot_p = np.dot(features, features.T)
        # _, diversity = np.linalg.slogdet(dot_p)

        # 方法2：计算最小生成树权重和（覆盖整个子集的最短路径）
        # 构建距离图
        graph = kneighbors_graph(features, n_neighbors=min(len(features) - 1, 20),
                                 mode='distance', include_self=False)
        # 计算最小生成树
        mst = minimum_spanning_tree(graph)
        diversity = mst.sum()

        return diversity

    def _calculate_objectives(self, individual):
        """计算两个目标函数值：不确定性和几何多样性"""
        # 1. 计算平均不确定性（最大化）
        uncertainty = np.mean(self.uncertainty_scores[individual])

        # 2. 计算几何多样性（最大化）
        diversity = self._calculate_geometric_diversity(individual)

        # 返回待最大化的目标值
        return [uncertainty, diversity]

    def _fast_non_dominated_sort(self, population):
        """执行快速非支配排序，返回各个支配前沿"""
        n = len(population)
        domination_count = [0] * n  # 被支配计数
        dominated_solutions = [[] for _ in range(n)]  # 支配的解集
        fronts = [[]]  # 存储各个前沿

        # 计算每个解的支配关系
        objectives = [self._calculate_objectives(ind) for ind in population]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # 检查i是否支配j
                if all(o_i >= o_j for o_i, o_j in zip(objectives[i], objectives[j])) and \
                        any(o_i > o_j for o_i, o_j in zip(objectives[i], objectives[j])):
                    # i支配j
                    dominated_solutions[i].append(j)
                # 检查j是否支配i
                elif all(o_j >= o_i for o_i, o_j in zip(objectives[i], objectives[j])) and \
                        any(o_j > o_i for o_i, o_j in zip(objectives[i], objectives[j])):
                    # j支配i
                    domination_count[i] += 1

            # 如果i不被任何解支配，将其添加到第一前沿
            if domination_count[i] == 0:
                fronts[0].append(i)

        # 生成所有前沿
        i = 0
        while i < len(fronts) and fronts[i]:  # 确保i在有效范围内
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
        """拥挤比较运算符：先比较等级，再比较拥挤度"""
        if rank[i] < rank[j]:
            return True
        elif rank[i] > rank[j]:
            return False
        elif distance[i] > distance[j]:  # 相同等级，比较拥挤度
            return True
        return False

    def _tournament_selection(self, population, rank, distance, k=2):
        """基于锦标赛选择的选择操作"""
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(range(len(population)), k)
            best = candidates[0]
            for candidate in candidates[1:]:
                if self._crowded_comparison_operator(candidate, best, rank, distance):
                    best = candidate
            selected.append(population[best].copy())
        return selected

    def _crossover(self, parent1, parent2):
        """执行交叉操作，确保结果不含重复元素"""
        if random.random() < self.crossover_rate:
            # 集合操作确保无重复
            combined = list(set(parent1).union(set(parent2)))
            if len(combined) <= self.budget:
                # 如果合并后元素不足，随机添加
                available_indices = list(set(range(self.n_samples)) - set(combined))
                if available_indices:  # 确保有可用索引
                    additional = random.sample(available_indices,
                                               min(self.budget - len(combined), len(available_indices)))
                    combined.extend(additional)

                # 如果仍然不足budget，用已有元素填充（可能有重复）
                while len(combined) < self.budget:
                    combined.append(random.choice(range(self.n_samples)))
            else:
                # 如果合并后元素过多，随机选择budget个
                combined = random.sample(combined, self.budget)
            return combined
        return parent1.copy()

    def _mutation(self, individual):
        """执行变异操作，确保结果不含重复元素"""
        individual = individual.copy()
        # 转为集合便于检查
        individual_set = set(individual)

        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                # 移除当前元素
                old_value = individual[i]
                individual_set.remove(old_value)

                # 生成一个不在individual_set中的随机索引
                tries = 0
                while tries < 10:  # 限制尝试次数
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
        """运行NSGA-II算法"""
        # 初始化种群
        population = self._initialize_population()

        # 主循环
        with tqdm(total=self.generations, desc="NSGA-II Optimization") as pbar:
            for _ in range(self.generations):
                # 生成子代
                offspring = []
                for _ in range(self.pop_size // 2):
                    # 选择两个父代
                    parents = random.sample(population, 2)
                    # 交叉
                    child1 = self._crossover(parents[0], parents[1])
                    child2 = self._crossover(parents[1], parents[0])
                    # 变异
                    child1 = self._mutation(child1)
                    child2 = self._mutation(child2)
                    offspring.extend([child1, child2])

                # 合并父代和子代
                combined = population + offspring

                try:
                    # 非支配排序
                    fronts, objectives = self._fast_non_dominated_sort(combined)

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

                    # 选择下一代种群
                    new_population = []
                    front_idx = 0
                    while len(new_population) + len(fronts[front_idx]) <= self.pop_size and front_idx < len(fronts):
                        # 完全添加当前前沿
                        for i in fronts[front_idx]:
                            new_population.append(combined[i])
                        front_idx += 1

                    # 如果还需要更多个体，根据拥挤度选择最后一个前沿的部分个体
                    if len(new_population) < self.pop_size and front_idx < len(fronts):
                        last_front = fronts[front_idx]
                        # 按拥挤度排序
                        sorted_last_front = sorted(last_front,
                                                   key=lambda i: crowding_distances.get(i, 0),
                                                   reverse=True)

                        # 添加所需数量的个体
                        remaining = self.pop_size - len(new_population)
                        for i in range(min(remaining, len(sorted_last_front))):
                            new_population.append(combined[sorted_last_front[i]])

                    # 如果仍然不足，随机添加（这是一个安全措施）
                    while len(new_population) < self.pop_size:
                        random_idx = random.randint(0, len(combined) - 1)
                        new_population.append(combined[random_idx])

                    # 更新种群
                    population = [ind.copy() for ind in new_population]

                except Exception as e:
                    print(f"迭代中发生错误: {str(e)}")
                    # 在出错时，保持当前种群不变，继续下一次迭代

                pbar.update(1)

        try:
            # 返回最后一代的非支配解
            final_fronts, final_objectives = self._fast_non_dominated_sort(population)

            if not final_fronts or not final_fronts[0]:
                print("警告：未找到任何帕累托最优解，返回随机解")
                return random.choice(population), [random.choice(population)], [[0, 0]]

            pareto_front = [population[i] for i in final_fronts[0]]
            pareto_objectives = [final_objectives[i] for i in final_fronts[0]]

            # 选择一个平衡的解返回
            # 这里选择不确定性和多样性乘积最大的解
            products = [obj[0] * obj[1] for obj in pareto_objectives]
            best_idx = np.argmax(products) if products else 0

            # 打印帕累托前沿信息
            print(f"发现{len(pareto_front)}个帕累托最优解")
            if pareto_objectives:
                print(
                    f"不确定性范围: [{min(obj[0] for obj in pareto_objectives):.4f}, {max(obj[0] for obj in pareto_objectives):.4f}]")
                print(
                    f"多样性范围: [{min(obj[1] for obj in pareto_objectives):.4f}, {max(obj[1] for obj in pareto_objectives):.4f}]")
                print(f"选择的解 - 不确定性: {pareto_objectives[best_idx][0]:.4f}, 多样性: {pareto_objectives[best_idx][1]:.4f}")

            return pareto_front[best_idx], pareto_front, pareto_objectives

        except Exception as e:
            print(f"计算最终结果时出错: {str(e)}")
            # 在出错时返回随机解
            return random.choice(population), [random.choice(population)], [[0, 0]]


if __name__ == "__main__":
    # 示例用法
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
