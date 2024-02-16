#python run_exp.py --model_tag codet5_small --task translate --sub_task cs-java
python run_exp.py --model_tag codet5_small --task translate --sub_task c-rust 2>&1 | tee run.log
