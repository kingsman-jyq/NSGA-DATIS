import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from time import time
from DATIS.DATIS import DATIS_test_input_selection, DATIS_redundancy_elimination
from DATIS.GA_DATIS import GADATIS
from mnist_test_selection import load_data, load_data_corrupted, calculate_rate, load_data_imdb
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

    rank_lst = DATIS_test_input_selection(
        softmax_prob, train_support, y_train, test_support, y_test, n_classes
    )

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

def calculate_apfd(test_case_order, fault_dict, total_tests):
    if not fault_dict:
        return 0.0

    m = len(fault_dict)
    n = total_tests

    first_positions = []

    for fault_id, fault_test_cases in fault_dict.items():
        for pos, test_idx in enumerate(test_case_order):
            if test_idx in fault_test_cases:
                first_positions.append(pos + 1)
                break
        else:
            first_positions.append(n + 1)

    sum_tf = sum(first_positions)
    apfd = 1 - (sum_tf / (n * m)) + (1 / (2 * n))

    return apfd

def get_fault_info(cluster_path):

    clustering_labels = np.load(cluster_path + '/cluster1.npy')
    mis_test_ind = np.load(cluster_path + '/mis_test_ind.npy')

    unique_faults = np.unique(clustering_labels)
    fault_dict = {}

    for fault_id in unique_faults:
        if fault_id == -1:
            indices = np.where(clustering_labels == -1)[0]
            fault_dict[-1] = [mis_test_ind[i] for i in indices]
        else:
            indices = np.where(clustering_labels == fault_id)[0]
            fault_dict[fault_id] = [mis_test_ind[i] for i in indices]


    total_faults = len(fault_dict)

    return fault_dict

def evaluate_methods(data_type, budget_ratios, dataset='mnist'):
    # 加载数据
    if dataset == 'mnist':
        if data_type == 'nominal':
            (x_train, y_train), (x_test, y_test) = load_data()
            cluster_path = './cluster_data/LeNet5_mnist_nominal'
        else:
            (x_train, y_train), _ = load_data()
            x_test, y_test = load_data_corrupted()
            cluster_path = './cluster_data/LeNet5_mnist_corrupted'

        model_path = "./model/model_mnist_LeNet5.hdf5"
        n_classes = 10
    else:
        (x_train, y_train), (x_test, y_test) = load_data_imdb()
        model_path = "./model/model_imdb_BiLstm.hdf5"
        cluster_path = "./cluster_data/BiLstm_imdb_nominal"
        n_classes = 2

    _, _, train_support, test_support, softmax_prob = load_model_and_features(
        model_path, x_train, x_test
    )

    rank_lst = DATIS_test_input_selection(
        softmax_prob, train_support, y_train, test_support, y_test, n_classes
    )
    uncertainty_scores = np.zeros(len(x_test))
    uncertainty_scores[rank_lst] = np.arange(len(rank_lst), 0, -1)

    datis_indices, datis_time = run_datis(
        softmax_prob, train_support, y_train, test_support, y_test, n_classes, budget_ratios
    )

    nsga_indices, nsga_time = run_nsga(
        test_support, uncertainty_scores, budget_ratios, x_test
    )

    # 评估故障检测率
    print(f"\nEvaluating {data_type} data...")
    print("\nDATIS Results:")
    datis_rates = calculate_rate(budget_ratios, test_support, x_test, rank_lst, datis_indices, cluster_path)

    print("\nNSGADATIS Results:")
    nsga_rates = calculate_rate(budget_ratios, test_support, x_test, rank_lst, nsga_indices, cluster_path)

    datis_apfd_scores = []
    nsga_apfd_scores = []
    total_tests = len(x_test)

    print("\nAPFD Results:")
    print("-------------")
    print(f"{'Budget Ratio':<15} {'DATIS APFD':<15} {'NSGA-II APFD':<15} {'Improvement':<15}")
    print("-" * 60)

    fault_dict = get_fault_info(cluster_path)

    for i, ratio in enumerate(budget_ratios):
        if datis_indices:
            datis_apfd = calculate_apfd(datis_indices[i], fault_dict, total_tests)
        else:
            k = int(total_tests * ratio)
            datis_apfd = calculate_apfd(rank_lst[:k], fault_dict, total_tests)

        nsga_apfd = calculate_apfd(nsga_indices[i], fault_dict, total_tests)

        datis_apfd_scores.append(datis_apfd)
        nsga_apfd_scores.append(nsga_apfd)

        improvement = ((nsga_apfd - datis_apfd) / datis_apfd) * 100 if datis_apfd > 0 else float('inf')

        print(f"{ratio:<15.4f} {datis_apfd:<15.4f} {nsga_apfd:<15.4f} {improvement:+.2f}%")

    return {
        'datis': {'time': datis_time, 'rates': datis_rates},
        'nsga': {'time': nsga_time, 'rates': nsga_rates}
    }

def plot_results(results_nominal, results_corrupted, budget_ratios):
    plt.figure(figsize=(15, 5))

    # 故障检测率比较
    plt.subplot(1, 2, 1)
    plt.plot(budget_ratios, results_nominal['datis']['rates'], 'b-o', label='DATIS (Nominal)')
    plt.plot(budget_ratios, results_nominal['nsga']['rates'], 'r--o', label='NSGADATIS (Nominal)')
    plt.plot(budget_ratios, results_corrupted['datis']['rates'], 'b-s', label='DATIS (Corrupted)')
    plt.plot(budget_ratios, results_corrupted['nsga']['rates'], 'r--s', label='NSGADATIS (Corrupted)')
    plt.xlabel('Budget Ratio')
    plt.ylabel('Fault Detection Rate')
    plt.title('Fault Detection Rate Comparison')
    plt.legend()
    plt.grid(True)

    # 执行时间比较
    plt.subplot(1, 2, 2)
    methods = ['DATIS', 'NSGADATIS']
    nominal_times = [results_nominal['datis']['time'], results_nominal['nsga']['time']]
    corrupted_times = [results_corrupted['datis']['time'], results_corrupted['nsga']['time']]

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
    budget_ratios = [0.001,0.002,0.003]

    results_nominal = evaluate_methods('nominal', budget_ratios)



