import numpy as np
import random
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer


class GADATIS:

    def __init__(self, test_features, uncertainty_scores, budget=500,
                 pop_size=50, generations=100, crossover_rate=0.8,
                 mutation_rate=0.01, alpha=0.7):
        """
            test_features: 测试集的潜在特征 [n_samples, feature_dim]
            uncertainty_scores: 测试输入的不确定性分数 [n_samples]
            budget: 需选择的测试输入数量
            pop_size: 种群大小
            generations: 迭代次数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            alpha: 适应度函数中不确定性的权重 (多样性权重=1-alpha)
        """
        self.test_features = Normalizer(norm='l2').fit_transform(test_features)  # L2归一化
        self.uncertainty_scores = uncertainty_scores
        self.budget = budget
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.n_samples = test_features.shape[0]

    def _initialize_population(self):
        return [random.sample(range(self.n_samples), self.budget)
                for _ in range(self.pop_size)]

    def _fitness_function(self, individual):
        # 1. 计算平均不确定性
        uncertainty = np.mean(self.uncertainty_scores[individual])

        # 2. 计算多样性（个体内所有测试输入间的平均距离）
        distances = []
        features = self.test_features[individual]
        # 使用KNN加速计算（仅需上三角距离）
        nbrs = NearestNeighbors(n_neighbors=len(features)-1, metric='euclidean').fit(features)
        distances, _ = nbrs.kneighbors(features)
        diversity = np.mean(distances)

        # 3. 加权综合
        return self.alpha * uncertainty + (1 - self.alpha) * diversity

    def _tournament_selection(self, population, fitness, k=3):
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(zip(population, fitness)), k)
            selected.append(max(candidates, key=lambda x: x[1])[0])
        return selected

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point1 = random.randint(1, self.budget - 2)
            point2 = random.randint(point1, self.budget - 1)
            child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            return child
        return parent1

    def _mutation(self, individual):
        for i in range(self.budget):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, self.n_samples - 1)
        return individual

    def _update_population(self, selected, offspring):
        # 更新种群：精英保留 + 锦标赛选择压缩
        # 合并父代和子代
        combined = selected + offspring

        # 评估适应度
        fitness = [self._fitness_function(ind) for ind in combined]

        elite_size = int(0.5 * self.pop_size)
        elite = [x for _, x in sorted(zip(fitness, combined), key=lambda x: -x[0])][:elite_size]

        # 剩余个体通过锦标赛选择
        non_elite = combined
        selected_non_elite = self._tournament_selection(
            non_elite, fitness, k=2
        )[:self.pop_size - elite_size]

        return elite + selected_non_elite

    def run(self):
        population = self._initialize_population()
        best_fitness = -np.inf
        best_individual = None

        with tqdm(total=self.generations, desc="GA Optimization") as pbar:
            for _ in range(self.generations):
                # 评估适应度
                fitness = [self._fitness_function(ind) for ind in population]

                # 记录当前最优
                current_best_idx = np.argmax(fitness)
                if fitness[current_best_idx] > best_fitness:
                    best_fitness = fitness[current_best_idx]
                    best_individual = population[current_best_idx]

                # 选择
                selected = self._tournament_selection(population, fitness)

                # 交叉与变异
                offspring = []
                for i in range(0, len(selected), 2):
                    if i + 1 < len(selected):
                        child1 = self._crossover(selected[i], selected[i + 1])
                        child2 = self._crossover(selected[i + 1], selected[i])
                        offspring.extend([self._mutation(child1), self._mutation(child2)])

                # 更新种群（保留精英）
                population = self._update_population(selected, offspring)
                pbar.update(1)

        return best_individual  # 返回最优解的测试输入索引


if __name__ == "__main__":
    n_samples = 10000
    feature_dim = 256
    test_features = np.random.rand(n_samples, feature_dim)  # 潜在特征
    uncertainty_scores = np.random.rand(n_samples)  # 不确定性分数

    # 初始化并运行GA
    ga = GADATIS(
        test_features=test_features,
        uncertainty_scores=uncertainty_scores,
        budget=500,  # 选择500个测试输入
        pop_size=50,  # 种群大小
        generations=100,  # 迭代次数
        alpha=0.7  # 不确定性权重
    )
    selected_indices = ga.run()
    print(f"Selected indices: {selected_indices[:10]}... (total {len(selected_indices)})")
