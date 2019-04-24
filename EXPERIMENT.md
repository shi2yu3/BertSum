#### ROUGE of original news articles
```
sudo rm -r results/*
cp fairseq_data/no_trunc/test.src.txt results/candidate
cp fairseq_data/no_trunc/test.tgt.txt results/gold
docker run --rm -it -v $(pwd):/workspace bertsum
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
exit
```
```
ROUGE-F(1/2/3/L): 14.92/8.52/10.86
ROUGE-R(1/2/3/l): 90.26/51.27/66.08
```


#### ROUGE of truncated news articles (max 400 tokens)
```
cp fairseq_data/trunc400/test.src.txt results/candidate
cp fairseq_data/trunc400/test.tgt.txt results/gold
docker run --rm -it -v $(pwd):/workspace bertsum
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
exit
```
```
ROUGE-F(1/2/3/L): 21.87/11.61/15.12
ROUGE-R(1/2/3/l): 83.86/44.74/58.69
```

#### ROUGE of lead baseline (author's data)
```
docker run --rm -it -v $(pwd):/workspace bertsum
python src/train.py -mode lead -encoder baseline -bert_data_path bert_data/downloaded -result_path results/cnndm -log_file logs/lead.log -report_rouge false
mv results/cnndm_step0.candidate results/candidate
mv results/cnndm_step0.gold results/gold
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
exit
```
```
ROUGE-F(1/2/3/L): 40.06/17.36/36.38
ROUGE-R(1/2/3/l): 51.99/22.64/47.17
```

#### ROUGE of lead baseline (regenerated data)

```
python src/train.py -mode lead -encoder baseline -bert_data_path bert_data/bert_base_uncased -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 40.08/17.39/36.39
ROUGE-R(1/2/3/l): 52.06/22.69/47.23
```

#### ROUGE of oracle baseline (author's data)
```
python src/train.py -mode oracle -encoder baseline -bert_data_path bert_data/downloaded -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 51.44/30.40/47.72
ROUGE-R(1/2/3/l): 52.32/30.77/48.50
```

#### ROUGE of oracle baseline (regenerated data)
```
python src/train.py -mode oracle -encoder baseline -bert_data_path bert_data/bert_base_uncased -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 51.37/30.35/47.65
ROUGE-R(1/2/3/l): 52.24/30.71/48.43
```

### Experimental results

To get ROUGE score of each experiment, run the following commands after the ```validate``` job finishs, whose Philly id is set to ```id```.
```
id=1555486458178_8522
sudo bash philly-fs.bash -ls //philly/eu2/ipgsrch/sys/jobs/application_$id | grep cnndm | grep candidate | awk '{print $NF}' | sed 's/cnndm_step//' | sed 's/\.candidate//' | nl | sed -e 's/^[[:space:]]*//' | while IFS=$'\t' read -r tgt src; do echo '\n\n$src'; sudo bash philly-fs.bash -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/cnndm_step$src.candidate results/cnndm$tgt.candidate; sudo bash philly-fs.bash -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/cnndm_step$src.gold results/cnndm$tgt.gold; done

docker run --rm -it -v $(pwd):/workspace bertsum
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
exit
```

##### Author's data
```
# train
python $rootdir/src/train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path $rootdir/bert_data/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000
python $rootdir/src/train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path $rootdir/bert_data/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
python $rootdir/src/train.py -mode train -encoder rnn -dropout 0.1 -bert_data_path $rootdir/bert_data/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -rnn_size 768

# validate
python $rootdir/src/train.py -mode validate -bert_data_path $rootdir/bert_data/cnndm -model_path ${modelpath}1555486458178_5127 -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $rootdir/bert_config_uncased_base.json -visible_gpus 0 -gpu_ranks 0 -batch_size 30000 -log_file $PHILLY_JOB_DIRECTORY/validate.log -result_path $PHILLY_JOB_DIRECTORY/cnndm -test_all -block_trigram true -report_rouge false
```
| train | validate | model | ROUGE-F(1/2/3/L) | ROUGE-R(1/2/3/l) |
| --- | --- | --- | --- | --- |
| [1555486458178_8574](https://philly/#/job/eu2/ipgsrch/1555486458178_8574) | [xxx](https://philly/#/job/eu2/ipgsrch/xxx) | classifier | xxx | xxx |
| [1555486458178_8575](https://philly/#/job/eu2/ipgsrch/1555486458178_8575) | [xxx](https://philly/#/job/eu2/ipgsrch/xxx) | transformer | xxx | xxx |
| [1555486458178_8576](https://philly/#/job/eu2/ipgsrch/1555486458178_8576) | [xxx](https://philly/#/job/eu2/ipgsrch/xxx) | rnn | xxx | xxx |



##### Regenerated data
```
# train
python $rootdir/src/train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path $rootdir/bert_data_regenerated/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000
python $rootdir/src/train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path $rootdir/bert_data_regenerated/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
python $rootdir/src/train.py -mode train -encoder rnn -dropout 0.1 -bert_data_path $rootdir/bert_data_regenerated/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -rnn_size 768

# validate
python $rootdir/src/train.py -mode validate -bert_data_path $rootdir/bert_data_regenerated/cnndm -model_path ${modelpath}1555486458178_7365/models -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $rootdir/bert_config_uncased_base.json -visible_gpus 0 -gpu_ranks 0 -batch_size 30000 -log_file $PHILLY_JOB_DIRECTORY/validate.log -result_path $PHILLY_JOB_DIRECTORY/cnndm -test_all -block_trigram true -report_rouge false
```
| train | validate | model | ROUGE-F(1/2/3/L) | ROUGE-R(1/2/3/l) |
| --- | --- | --- | --- | --- |
| [1555486458178_8590](https://philly/#/job/eu2/ipgsrch/1555486458178_8590) | [1555486458178_8522xxx](https://philly/#/job/eu2/ipgsrch/1555486458178_8522) | classifier | 42.82/20.03/39.28 | 52.67/24.61/48.27 |
| [1555486458178_8591](https://philly/#/job/eu2/ipgsrch/1555486458178_8591) | [1555486458178_8523xxx](https://philly/#/job/eu2/ipgsrch/1555486458178_8523) | transformer | 42.85/20.09/39.32 | 52.69/24.66/48.28 |
| [1555486458178_8592](https://philly/#/job/eu2/ipgsrch/1555486458178_8592) | [1555486458178_8524xxx](https://philly/#/job/eu2/ipgsrch/1555486458178_8524) | rnn | ? | --- |
