from others.utils import test_rouge, rouge_results_to_str

can_path = "results/cnndm.candidate"
gold_path = "results/cnndm.gold"
rouges = test_rouge("results", can_path, gold_path)
print('Rouges: \n%s' % (rouge_results_to_str(rouges)))
