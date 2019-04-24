#### ROUGE of original article
```
cp ../cnndm/fairseq_data_no_truncate/test.src.txt results/cnndm.candidate
cp ../cnndm/fairseq_data_no_truncate/test.tgt.txt results/cnndm.gold
docker run --rm -it -v $(pwd):/workspace bertsum
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
exit
```
```
ROUGE-F(1/2/3/L): 14.92/8.52/10.86
ROUGE-R(1/2/3/l): 90.26/51.27/66.08
```


#### ROUGE of truncated article (max 400 tokens)
```
cp ../cnndm/fairseq_data/test.src.txt results/cnndm.candidate
cp ../cnndm/fairseq_data/test.tgt.txt results/cnndm.gold
docker run --rm -it -v $(pwd):/workspace bertsum
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
exit
```
```
ROUGE-F(1/2/3/L): 21.87/11.61/15.12
ROUGE-R(1/2/3/l): 83.86/44.74/58.69
```

#### ROUGE of lead (downloaded data)
```
docker run --rm -it -v $(pwd):/workspace bertsum
cd BertSum
python src/train.py -mode lead -encoder baseline -bert_data_path ../cnndm/bert_data/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
mv results/cnndm_step0.candidate results/cnndm.candidate
mv results/cnndm_step0.gold results/cnndm.gold
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
```
```
ROUGE-F(1/2/3/L): 40.06/17.36/36.38
ROUGE-R(1/2/3/l): 51.99/22.64/47.17
```

#### ROUGE of lead (regenerated data)
```
python src/train.py -mode lead -encoder baseline -bert_data_path ../cnndm/bert_data_regenerated/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 40.08/17.39/36.39
ROUGE-R(1/2/3/l): 52.06/22.69/47.23
```


#### ROUGE of oracle (downloaded data)
```
python src/train.py -mode oracle -encoder baseline -bert_data_path ../cnndm/bert_data/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 51.44/30.40/47.72
ROUGE-R(1/2/3/l): 52.32/30.77/48.50
```

#### ROUGE of oracle (regenerated data)
```
python src/train.py -mode oracle -encoder baseline -bert_data_path ../cnndm/bert_data_regenerated/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 51.37/30.35/47.65
ROUGE-R(1/2/3/l): 52.24/30.71/48.43
```

#### Experimental results

##### Downloaded data
```
# train
python $rootdir/src/train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path $rootdir/bert_data/cnndm -model_path $PHILLY_JOB_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000
python $rootdir/src/train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path $rootdir/bert_data/cnndm -model_path $PHILLY_JOB_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
python $rootdir/src/train.py -mode train -encoder rnn -dropout 0.1 -bert_data_path $rootdir/bert_data/cnndm -model_path $PHILLY_JOB_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -rnn_size 768
# validate
python $rootdir/src/train.py -mode validate -bert_data_path $rootdir/bert_data/cnndm -model_path ${modelpath}1555486458178_5127 -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $rootdir/bert_config_uncased_base.json -visible_gpus 0 -gpu_ranks 0 -batch_size 30000 -log_file $PHILLY_JOB_DIRECTORY/validate.log -result_path $PHILLY_JOB_DIRECTORY/cnndm -test_all -block_trigram true -report_rouge false
# test
python $rootdir/src/train.py -mode test -bert_data_path $rootdir/bert_data/cnndm -model_path ${modelpath}1555486458178_5127 -test_from ${modelpath}1555486458178_5127/model_step_50000.pt -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $rootdir/bert_config_uncased_base.json -visible_gpus 0 -gpu_ranks 0 -batch_size 30000 -log_file $PHILLY_JOB_DIRECTORY/test.log -result_path $PHILLY_JOB_DIRECTORY/cnndm -block_trigram true -report_rouge false
```
**rouge**
```
id=1555486458178_8521
sudo bash /home/yushi/philly-fs.bash -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/cnndm_step50000.candidate results/cnndm.candidate
sudo bash /home/yushi/philly-fs.bash -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/cnndm_step50000.gold results/cnndm.gold
docker run --rm -it -v $(pwd):/workspace bertsum
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
```
| train | validate | test | model | data | ROUGE-F(1/2/3/L) | ROUGE-R(1/2/3/l) |
| --- | --- | --- | --- | --- | --- | --- |
| [1555486458178_5127](https://philly/#/job/eu2/ipgsrch/1555486458178_5127) | [1555486458178_8506](https://philly/#/job/eu2/ipgsrch/1555486458178_8506) | [1555486458178_7344](https://philly/#/job/eu2/ipgsrch/1555486458178_7344) | classifier | downloaded | 42.91/20.04/39.32 | 53.43/24.92/48.91 |
| [1555486458178_5623](https://philly/#/job/eu2/ipgsrch/1555486458178_5623) | [1555486458178_8507](https://philly/#/job/eu2/ipgsrch/1555486458178_8507) | [1555486458178_8519](https://philly/#/job/eu2/ipgsrch/1555486458178_8519) | transformer | downloaded | 42.71/19.88/39.13 | 53.48/24.88/48.95 |
| [1555486458178_5938](https://philly/#/job/eu2/ipgsrch/1555486458178_5938) | [1555486458178_8508](https://philly/#/job/eu2/ipgsrch/1555486458178_8508) | [1555486458178_8521](https://philly/#/job/eu2/ipgsrch/1555486458178_8521) | rnn | downloaded | 42.84/20.00/39.26 | 53.43/24.89/48.91 |



##### Regenerated data
```
# train
python $rootdir/src/train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path $rootdir/bert_data_regenerated/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 5000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000
python $rootdir/src/train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path $rootdir/bert_data_regenerated/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 5000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
python $rootdir/src/train.py -mode train -encoder rnn -dropout 0.1 -bert_data_path $rootdir/bert_data_regenerated/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 5000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -rnn_size 768
# validate
python $rootdir/src/train.py -mode validate -bert_data_path $rootdir/bert_data_regenerated/cnndm -model_path ${modelpath}1555486458178_7365/models -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $rootdir/bert_config_uncased_base.json -visible_gpus 0 -gpu_ranks 0 -batch_size 30000 -log_file $PHILLY_JOB_DIRECTORY/validate.log -result_path $PHILLY_JOB_DIRECTORY/cnndm -test_all -block_trigram true -report_rouge false
# test
python $rootdir/src/train.py -mode test -bert_data_path $rootdir/bert_data/cnndm -model_path ${modelpath}1555486458178_7367/models -test_from ${modelpath}1555486458178_7367/models/model_step_50000.pt -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $rootdir/bert_config_uncased_base.json -visible_gpus 0 -gpu_ranks 0 -batch_size 30000 -log_file $PHILLY_JOB_DIRECTORY/test.log -result_path $PHILLY_JOB_DIRECTORY/cnndm -block_trigram true -report_rouge false
```
| train | validate | test | model | data | ROUGE-F(1/2/3/L) | ROUGE-R(1/2/3/l) |
| --- | --- | --- | --- | --- | --- | --- |
| [1555486458178_7365](https://philly/#/job/eu2/ipgsrch/1555486458178_7365) | [1555486458178_8522](https://philly/#/job/eu2/ipgsrch/1555486458178_8522) | [1555486458178_8527](https://philly/#/job/eu2/ipgsrch/1555486458178_8527) | classifier | generated | 42.82/20.03/39.28 | 52.67/24.61/48.27 |
| [1555486458178_7367](https://philly/#/job/eu2/ipgsrch/1555486458178_7367) | [1555486458178_8523](https://philly/#/job/eu2/ipgsrch/1555486458178_8523) | [1555486458178_8526](https://philly/#/job/eu2/ipgsrch/1555486458178_8526) | transformer | generated | 42.85/20.09/39.32 | 52.69/24.66/48.28 |
| [1555486458178_7368](https://philly/#/job/eu2/ipgsrch/1555486458178_7368) | [1555486458178_8524](https://philly/#/job/eu2/ipgsrch/1555486458178_8524) | [1555486458178_8528](https://philly/#/job/eu2/ipgsrch/1555486458178_8528) | rnn | generated | ? | --- |
