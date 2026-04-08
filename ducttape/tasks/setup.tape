task CloneSimpleEvals
    > simple_evals_repo
{
    git clone https://github.com/zeppombal/simple-evals.git
    mv simple-evals simple_evals_repo
    wget https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl -O simple_evals_repo/hb_data/meta_eval.jsonl
}

task SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    > healthbench_env
    :: meta_venv=@
{
    source $meta_venv/bin/activate
    python -m venv healthbench_env
    source healthbench_env/bin/activate
    pip install -r $evals_repo/requirements.txt
}

task SetUpIFEvalEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    > ifeval_env
    :: meta_venv=@
{
    source $meta_venv/bin/activate
    python -m venv ifeval_env
    source ifeval_env/bin/activate
    pip install -r $evals_repo/instruction_following_eval/requirements.txt
}
