from others.utils import test_rouge, rouge_results_to_str
import os

rouges = {}
sum = {}
n = 0
for i in ["", "1", "2", "3"]:
    can_path = f'results/candidate{i}'
    gold_path = f'results/gold{i}'
    if os.path.exists(can_path) and os.path.exists(gold_path):
        print(f'{"*" * 40}\n{can_path} <--> {gold_path}')
        rouges[i] = test_rouge("results", can_path, gold_path)
        if i is not "":
            for key in rouges[i]:
                if key in sum:
                    sum[key] += rouges[i][key]
                else:
                    sum[key] = rouges[i][key]
            n += 1
        print(f'Rouges of results/candidate{i}: \n{rouge_results_to_str(rouges[i])}')

print('*' * 10 + ' Summary ' + '*' * 10)

for key in rouges:
    print(f'Rouges of results/candidate{key}: \n{rouge_results_to_str(rouges[key])}')

if n > 0:
    print(f'Sum of rouges: \n{rouge_results_to_str(sum)}')
    for key in sum:
        sum[key] /= n
    print(f'Average of {n} rouges: \n{rouge_results_to_str(sum)}')
