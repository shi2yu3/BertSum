# Install pyrouge
```
git clone https://github.com/andersjo/pyrouge
cd pyrouge/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
cd ../../../..
```

**Option 1**: only install pyrouge
```
apt-get update && apt-get install -y perl synaptic
pip install pyrouge
```

**Option 2**: build docker image
```
docker build -t bertsum .
```

# Some baseline numbers

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
python src/train.py -mode lead -encoder baseline -bert_data_path bert_data/downloaded/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
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

#### ROUGE of lead baseline (regenerated bert_base_uncased data)

```
python src/train.py -mode lead -encoder baseline -bert_data_path bert_data/bert_base_uncased/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 40.08/17.39/36.39
ROUGE-R(1/2/3/l): 52.06/22.69/47.23
```

#### ROUGE of lead baseline (regenerated base_large_uncased data)

```
python src/train.py -mode lead -encoder baseline -bert_data_path bert_data/bert_large_uncased/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 40.08/17.39/36.39
ROUGE-R(1/2/3/l): 52.06/22.69/47.23
```

#### ROUGE of oracle baseline (author's data)
```
python src/train.py -mode oracle -encoder baseline -bert_data_path bert_data/downloaded/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 51.44/30.40/47.72
ROUGE-R(1/2/3/l): 52.32/30.77/48.50
```

#### ROUGE of oracle baseline (regenerated bert_base_uncased data)
```
python src/train.py -mode oracle -encoder baseline -bert_data_path bert_data/bert_base_uncased/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 51.37/30.35/47.65
ROUGE-R(1/2/3/l): 52.24/30.71/48.43
```

#### ROUGE of oracle baseline (regenerated bert_large_uncased data)
```
python src/train.py -mode oracle -encoder baseline -bert_data_path bert_data/bert_large_uncased/cnndm -result_path results/cnndm -log_file logs/lead.log -report_rouge false
```
```
ROUGE-F(1/2/3/L): 51.37/30.35/47.65
ROUGE-R(1/2/3/l): 52.24/30.71/48.43
```

# Experiment setup

* Upload data and code to Philly
```
sudo bash philly-fs.bash -cp -r bert_data/downloaded/ //philly/eu2/ipgsrch/yushi/bertsum/bert_data/
sudo bash philly-fs.bash -cp -r bert_data/bert_base_uncased/ //philly/eu2/ipgsrch/yushi/bertsum/bert_data/
sudo bash philly-fs.bash -cp -r bert_data/bert_large_uncased/ //philly/eu2/ipgsrch/yushi/bertsum/bert_data/
```
* Prepare ```job.json``` for Philly job submission.
* To submit ```job.json``` to Philly, use command 
```
curl --ntlm --user : -X POST -H "Content-Type: application/json" --data @job.json https://philly/api/jobs
```
* To get ROUGE score of each experiment, run the following commands after the ```validate``` job finishs, whose Philly id is set to ```id```.
```
phillyfs=philly-fs.bash
id=1555486458178_11502	
sudo bash $phillyfs -ls //philly/eu2/ipgsrch/sys/jobs/application_$id | grep cnndm | grep candidate | awk '{print $NF}' | sed 's/cnndm_step//' | sed 's/\.candidate//' | nl | sed -e 's/^[[:space:]]*//' | while IFS=$'\t' read -r tgt src; do echo $src; sudo bash $phillyfs -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/cnndm_step$src.candidate results/candidate$tgt; sudo bash $phillyfs -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/cnndm_step$src.gold results/gold$tgt; done

docker run --rm -it -v $(pwd):/workspace bertsum
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
exit
```


# Experimental results


This section gives both Philly ```commandLine``` in ```job.json``` and ROUGE scores.

##### Author's data

Add ```"bertdata": "downloaded"``` in ```environmentVariables``` of ```job.json```. Copy one of the following command in ```commandLine``` of ```job.json```.
```
# bertdata in environmentVariables
"bertdata": "downloaded"

# train commandLine
python $rootdir/src/train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path $rootdir/bert_data/$bertdata/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000
python $rootdir/src/train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path $rootdir/bert_data/$bertdata/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
python $rootdir/src/train.py -mode train -encoder rnn -dropout 0.1 -bert_data_path $rootdir/bert_data/$bertdata/cnndm -model_path $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -bert_config_path $PHILLY_JOB_DIRECTORY/bert_config_uncased_base.json -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file $PHILLY_JOB_DIRECTORY/train.log -use_interval true -warmup_steps 10000 -rnn_size 768

# validate commandLine
python $rootdir/src/train.py -mode validate -bert_data_path $rootdir/bert_data/$bertdata/cnndm -model_path ${modelpath}_1555486458178_8574/models -tensorboard_log_dir $PHILLY_MODEL_DIRECTORY -temp_dir $PHILLY_JOB_DIRECTORY -bert_config_path $rootdir/bert_config_uncased_base.json -visible_gpus 0 -gpu_ranks 0 -batch_size 30000 -log_file $PHILLY_JOB_DIRECTORY/validate.log -result_path $PHILLY_JOB_DIRECTORY/cnndm -test_all -block_trigram true -report_rouge false
```
| train | validate | model | ROUGE-F(1/2/3/L) | ROUGE-R(1/2/3/l) |
| --- | --- | --- | --- | --- |
| [1555486458178_8574](https://philly/#/job/eu2/ipgsrch/1555486458178_8574) | [1555486458178_10286](https://philly/#/job/eu2/ipgsrch/1555486458178_10286) | classifier | 42.87/20.05/39.28 | 53.60/25.04/49.06 |
| [1555486458178_8575](https://philly/#/job/eu2/ipgsrch/1555486458178_8575) | [1555486458178_11502](https://philly/#/job/eu2/ipgsrch/1555486458178_11502) | transformer | 42.93/20.11/39.38 | 53.33/24.96/48.86 |
| [1555486458178_8576](https://philly/#/job/eu2/ipgsrch/1555486458178_8576) | [1555486458178_11504](https://philly/#/job/eu2/ipgsrch/1555486458178_11504) | rnn | 42.96/20.12/39.40 | 53.29/24.92/48.82 |



##### Regenerated data

Set ```"bertdata": "bert_base_uncased"``` in ```environmentVariables``` of ```job.json```.

| train | validate | model | ROUGE-F(1/2/3/L) | ROUGE-R(1/2/3/l) |
| --- | --- | --- | --- | --- |
| [1555486458178_8590](https://philly/#/job/eu2/ipgsrch/1555486458178_8590) | [1555486458178_11505](https://philly/#/job/eu2/ipgsrch/1555486458178_11505) | classifier | 42.76/20.01/39.22 | 53.14/24.83/48.68 |
| [1555486458178_8591](https://philly/#/job/eu2/ipgsrch/1555486458178_8591) | [1555486458178_11506](https://philly/#/job/eu2/ipgsrch/1555486458178_11506) | transformer | 42.81/20.06/39.26 | 53.21/24.90/48.74 |
| [1555486458178_8592](https://philly/#/job/eu2/ipgsrch/1555486458178_8592) | [1555486458178_11507](https://philly/#/job/eu2/ipgsrch/1555486458178_11507) | rnn | 42.78/20.04/39.23 | 53.34/24.98/48.86 |
