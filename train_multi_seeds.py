import numpy as np
from main import main
from configs.basic_config import get_args

def calculate_statistics(all_results):
    stats = {}
    metrics = ['c_acc', 'y_acc', 'c_auc', 'y_auc']

    for metric in metrics:
        values = [float(result['model_results'][metric].strip('%')) for result in all_results]
        mean = np.mean(values)
        max_dev = max(abs(max(values) - mean), abs(min(values) - mean))
        stats[metric] = {
            'mean': f"{mean:.2f}%",
            'interval': f"±{max_dev:.2f}%",
            'raw_values': values
        }
    return stats

if __name__ == '__main__':
    args = get_args()
    seeds = [42, 2024, 2025, 1, 2]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*20} Running {args.project_name} on dataset {args.dataset} - Seed {seed} {'='*20}")
        result = main(seed=seed, args=args)
        all_results.append(result)

    statistics = calculate_statistics(all_results)
    print("\nFinal statistics:")
    for metric, stats in statistics.items():
        print(f"{metric}: {stats['mean']} {stats['interval']}")
