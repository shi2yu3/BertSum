**job.json**
```
{
  "version": "2019_04_03",
  "metadata": {
    "name": "bertsum",
    "cluster": "eu2",
    "vc": "ipgsrch",
    "username": "yushi"
  },
  "environmentVariables": {
    "rootdir": "/philly/eu2/ipgsrch/yushi/bertsum"
  },
  "resources": {
    "workers": {
      "type": "skuResource",
      "sku": "G1",
      "count": 1,
      "image": "phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch1.0.0-py36-vcr",
      "commandLine": "python $rootdir/src/train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path $rootdir/bert_data/cnndm -model_path $PHILLY_JOB_DIRECTORY -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 1 -log_file $PHILLY_JOB_DIRECTORY -use_interval true -warmup_steps 10000"
    }
  }
}
```

**submission**
```
curl --ntlm --user : -X POST -H "Content-Type: application/json" --data @job.json https://philly/api/jobs
```