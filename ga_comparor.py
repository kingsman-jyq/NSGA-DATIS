import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from time import time
from DATIS.DATIS import DATIS_test_input_selection, DATIS_redundancy_elimination
from DATIS.GA_DATIS import GADATIS
from mnist_test_selection import load_data, load_data_corrupted, calculate_rate
from mnist_dnn_enhancement import retrain
from DATIS.NSGAII_DATIS import NSGADATIS


def load_model_and_features(model_path, x_train, x_test):
    ori_model = load_model(model_path)
    new_model = Model(ori_model.input, outputs=ori_model.layers[-2].output)

    train_support_output = new_model.predict(x_train)
    train_support_output = np.squeeze(train_support_output)
    test_support_output = new_model.predict(x_test)
    test_support_output = np.squeeze(test_support_output)
    softmax_test_prob = ori_model.predict(x_test)

    return new_model, ori_model, train_support_output, test_support_output, softmax_test_prob


def run_datis(softmax_prob, train_support, y_train, test_support, y_test, n_classes, budget_ratios):
    start_time = time()

    # 第一阶段：测试输入选择
    rank_lst = DATIS_test_input_selection(
        softmax_prob, train_support, y_train, test_support, y_test, n_classes
    )

    # 第二阶段：冗余消除
    selected_indices = DATIS_redundancy_elimination(
        budget_ratios, rank_lst, test_support, y_test
    )

    elapsed_time = time() - start_time
    return selected_indices, elapsed_time


def run_gadatis(test_support, uncertainty_scores, budget_ratios, x_test):
    start_time = time()

    budgets = [int(len(x_test) * ratio) for ratio in budget_ratios]

    selected_indices = []
    for budget in budgets:
        # 初始化并运行遗传算法
        ga = GADATIS(
            test_features=test_support,
            uncertainty_scores=uncertainty_scores,
            budget=budget,
            pop_size=50,
            generations=50,
            alpha=0.7
        )
        indices = ga.run()
        selected_indices.append(indices)

    elapsed_time = time() - start_time
    return selected_indices, elapsed_time

def run_nsga(test_support, uncertainty_scores, budget_ratios, x_test):
    start_time = time()
    budgets = [int(len(x_test) * ratio) for ratio in budget_ratios]
    selected_indices = []
    for budget in budgets:
        nsga = NSGADATIS(
            test_features=test_support,
            uncertainty_scores=uncertainty_scores,
            budget=budget,
            pop_size=100,
            generations=100,
        )
        indices, _, _ = nsga.run()
        selected_indices.append(indices)

    elapsed_time = time() - start_time
    return selected_indices, elapsed_time

def evaluate_methods(data_type, budget_ratios):
    # 加载数据
    if data_type == 'nominal':
        (x_train, y_train), (x_test, y_test) = load_data()
        cluster_path = './cluster_data/LeNet5_mnist_nominal'
    else:
        (x_train, y_train), _ = load_data()
        x_test, y_test = load_data_corrupted()
        cluster_path = './cluster_data/LeNet5_mnist_corrupted'

    model_path = "./model/model_mnist_LeNet5.hdf5"
    n_classes = 10

    # 提取特征
    _, _, train_support, test_support, softmax_prob = load_model_and_features(
        model_path, x_train, x_test
    )

    # 计算不确定性分数（DATIS第一阶段的输出）
    rank_lst = DATIS_test_input_selection(
        softmax_prob, train_support, y_train, test_support, y_test, n_classes
    )
    uncertainty_scores = np.zeros(len(x_test))
    uncertainty_scores[rank_lst] = np.arange(len(rank_lst), 0, -1)

    datis_indices, datis_time = run_datis(
        softmax_prob, train_support, y_train, test_support, y_test, n_classes, budget_ratios
    )

    # gadatis_indices, gadatis_time = run_gadatis(
    #     test_support, uncertainty_scores, budget_ratios, x_test
    # )

    nsga_indices, nsga_time = run_nsga(
        test_support, uncertainty_scores, budget_ratios, x_test
    )

    # 评估故障检测率
    print(f"\nEvaluating {data_type} data...")
    print("\nDATIS Results:")
    datis_rates = calculate_rate(budget_ratios, test_support, x_test, rank_lst, datis_indices, cluster_path)

    # print("\nGADATIS Results:")
    # gadatis_rates = calculate_rate(budget_ratios, test_support, x_test, rank_lst, gadatis_indices, cluster_path)

    print("\nNSGADATIS Results:")
    nsga_rates = calculate_rate(budget_ratios, test_support, x_test, rank_lst, nsga_indices, cluster_path)
    # 评估DNN增强效果
    mid_index = len(x_test) // 2
    x_val = x_test[mid_index:]
    y_val = y_test[mid_index:]

    print("\nDNN Enhancement Evaluation:")
    for i, ratio in enumerate(budget_ratios):
        print(f"\nBudget ratio: {ratio}")

        # DATIS增强
        x_s_datis = x_test[datis_indices[i]]
        y_s_datis = y_test[datis_indices[i]]
        print("DATIS enhancement:")
        retrain(data_type, model_path, x_s_datis, y_s_datis, x_train, y_train, x_val, y_val, n_classes)

        # GADATIS增强
        # x_s_ga = x_test[gadatis_indices[i]]
        # y_s_ga = y_test[gadatis_indices[i]]
        # print("GADATIS enhancement:")
        # retrain(data_type, model_path, x_s_ga, y_s_ga, x_train, y_train, x_val, y_val, n_classes)

        # NSGADATIS增强
        x_s_nsga = x_test[nsga_indices[i]]
        y_s_nsga = y_test[nsga_indices[i]]
        print("NSGADATIS enhancement:")
        retrain(data_type, model_path, x_s_nsga, y_s_nsga, x_train, y_train, x_val, y_val, n_classes)

    return {
        'datis': {'time': datis_time, 'rates': datis_rates},
        # 'gadatis': {'time': gadatis_time, 'rates': gadatis_rates},
        'nsga': {'time': nsga_time, 'rates': nsga_rates}
    }


def plot_results(results_nominal, results_corrupted, budget_ratios):
    """可视化比较结果"""
    plt.figure(figsize=(15, 5))

    # 故障检测率比较
    plt.subplot(1, 2, 1)
    plt.plot(budget_ratios, results_nominal['datis']['rates'], 'b-o', label='DATIS (Nominal)')
    plt.plot(budget_ratios, results_nominal['gadatis']['rates'], 'r--o', label='GADATIS (Nominal)')
    plt.plot(budget_ratios, results_corrupted['datis']['rates'], 'b-s', label='DATIS (Corrupted)')
    plt.plot(budget_ratios, results_corrupted['gadatis']['rates'], 'r--s', label='GADATIS (Corrupted)')
    plt.xlabel('Budget Ratio')
    plt.ylabel('Fault Detection Rate')
    plt.title('Fault Detection Rate Comparison')
    plt.legend()
    plt.grid(True)

    # 执行时间比较
    plt.subplot(1, 2, 2)
    methods = ['DATIS', 'GADATIS']
    nominal_times = [results_nominal['datis']['time'], results_nominal['gadatis']['time']]
    corrupted_times = [results_corrupted['datis']['time'], results_corrupted['gadatis']['time']]

    x = np.arange(len(methods))
    width = 0.35

    plt.bar(x - width / 2, nominal_times, width, label='Nominal')
    plt.bar(x + width / 2, corrupted_times, width, label='Corrupted')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time Comparison')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.show()


if __name__ == '__main__':
    # 实验设置
    budget_ratios = [0.001, 0.005, 0.01, 0.05, 0.1]

    results_nominal = evaluate_methods('nominal', budget_ratios)
    results_corrupted = evaluate_methods('corrupted', budget_ratios)

    # 可视化结果
    # plot_results(results_nominal, results_corrupted, budget_ratios)

