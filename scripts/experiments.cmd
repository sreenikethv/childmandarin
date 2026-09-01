getenv = True

Requirements = (machine == "patas-gn3.ling.washington.edu")

executable = /projects/assigned/ChildLang/finetune_fairseq/scripts/run_experiments.sh

arguments = /projects/assigned/ChildLang/finetune_fairseq/configs/$(config)

output = /projects/assigned/ChildLang/finetune_fairseq/logs/$(experiment)/$(experiment).out
error  = /projects/assigned/ChildLang/finetune_fairseq/logs/$(experiment)/$(experiment).err
log    = /projects/assigned/ChildLang/finetune_fairseq/logs/$(experiment)/$(experiment).log

+Research = true

request_memory = 2048
request_gpus = 1

notification = Complete

queue experiment, config from /projects/assigned/ChildLang/finetune_fairseq/configs/config_list.txt