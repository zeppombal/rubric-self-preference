import generation.tape

task EvaluateHealthBench
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    < generations_path=$generations_path@GenerateHealthBench
    > evaluations_path=evaluation.json
    :: evaluator_vllm_serve_str=@
    :: evaluator_completion_args=@
    :: evaluator_url=@
    :: evaluator=@
    :: vllm_env=@
    :: max_workers=@
{
    # init vllm if evaluator_vllm_serve_str is not empty
    wait_for_server() {
        local max_attempts=18000  # 5 hours timeout
        local attempt=0
        
        echo "Waiting for vLLM server to start..."
        while [ $attempt -lt $max_attempts ]; do
            if curl --silent --fail "${evaluator_url}/models" > /dev/null 2>&1; then
                echo "vLLM server is ready!"
                return 0
            fi
            
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "vLLM server process died unexpectedly"
                return 1
            fi
            
            ((attempt++))
            echo "Attempt $attempt/$max_attempts: vLLM server not ready yet, retrying in 1 second..."
            sleep 1
        done
        
        echo "Timeout waiting for vLLM server"
        return 1
    }
    if [ -n "$evaluator_vllm_serve_str" ]; then
        sleep 60 # wait a minute for vllm cleanup from previous tasks
        source $vllm_env/bin/activate
        vllm serve $evaluator_vllm_serve_str > vllm.log 2> vllm_error.log &
        VLLM_PID=$!
        if ! wait_for_server; then
            exit 1
        fi
    fi

    # run evaluations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python simple_evals.py \
        --eval=healthbench \
        --mode=evaluate \
        --grader-model=$evaluator \
        --grader-completion-args=$evaluator_completion_args \
        --n-threads=$max_workers \
        --generation-input=$generations_path \
        --output=$evaluations_path \
        --output-dir="$(dirname $evaluations_path)"
    
    # kill vllm server
    if [ -n "$evaluator_vllm_serve_str" ]; then
    kill $VLLM_PID
    fi
}

task EvaluateIFEval
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    < generations_path=$generations_path@GenerateIFEval
    > evaluations_path=evaluation.json
    :: evaluator_vllm_serve_str=@
    :: evaluator_completion_args=@
    :: evaluator_url=@
    :: evaluator=@
    :: vllm_env=@
    :: max_workers=@
{
    # init vllm if evaluator_vllm_serve_str is not empty
    wait_for_server() {
        local max_attempts=18000  # 5 hours timeout
        local attempt=0
        
        echo "Waiting for vLLM server to start..."
        while [ $attempt -lt $max_attempts ]; do
            if curl --silent --fail "${evaluator_url}/models" > /dev/null 2>&1; then
                echo "vLLM server is ready!"
                return 0
            fi
            
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "vLLM server process died unexpectedly"
                return 1
            fi
            
            ((attempt++))
            echo "Attempt $attempt/$max_attempts: vLLM server not ready yet, retrying in 1 second..."
            sleep 1
        done
        
        echo "Timeout waiting for vLLM server"
        return 1
    }
    if [ -n "$evaluator_vllm_serve_str" ]; then
        sleep 60 # wait a minute for vllm cleanup from previous tasks
        source $vllm_env/bin/activate
        vllm serve $evaluator_vllm_serve_str > vllm.log 2> vllm_error.log &
        VLLM_PID=$!
        if ! wait_for_server; then
            exit 1
        fi
    fi

    # run evaluations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python simple_evals.py \
        --eval=ifeval \
        --mode=evaluate \
        --grader-model=$evaluator \
        --grader-completion-args=$evaluator_completion_args \
        --n-threads=$max_workers \
        --generation-input=$generations_path \
        --output=$evaluations_path \
        --output-dir="$(dirname $evaluations_path)"
    
    # kill vllm server
    if [ -n "$evaluator_vllm_serve_str" ]; then
    kill $VLLM_PID
    fi
}

task EvaluateIFEvalAR
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    < generations_path=$generations_path@GenerateIFEval
    > evaluations_path=evaluation.json
    :: evaluator_vllm_serve_str=@
    :: evaluator_completion_args=@
    :: evaluator_url=@
    :: evaluator=@
    :: vllm_env=@
    :: max_workers=@
{
    # init vllm if evaluator_vllm_serve_str is not empty
    wait_for_server() {
        local max_attempts=18000  # 5 hours timeout
        local attempt=0
        
        echo "Waiting for vLLM server to start..."
        while [ $attempt -lt $max_attempts ]; do
            if curl --silent --fail "${evaluator_url}/models" > /dev/null 2>&1; then
                echo "vLLM server is ready!"
                return 0
            fi
            
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "vLLM server process died unexpectedly"
                return 1
            fi
            
            ((attempt++))
            echo "Attempt $attempt/$max_attempts: vLLM server not ready yet, retrying in 1 second..."
            sleep 1
        done
        
        echo "Timeout waiting for vLLM server"
        return 1
    }
    if [ -n "$evaluator_vllm_serve_str" ]; then
        sleep 60 # wait a minute for vllm cleanup from previous tasks
        source $vllm_env/bin/activate
        vllm serve $evaluator_vllm_serve_str > vllm.log 2> vllm_error.log &
        VLLM_PID=$!
        if ! wait_for_server; then
            exit 1
        fi
    fi

    # run evaluations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python simple_evals.py \
        --eval=ifeval_ar \
        --mode=evaluate \
        --grader-model=$evaluator \
        --grader-completion-args=$evaluator_completion_args \
        --n-threads=$max_workers \
        --generation-input=$generations_path \
        --output=$evaluations_path \
        --output-dir="$(dirname $evaluations_path)"
    
    # kill vllm server
    if [ -n "$evaluator_vllm_serve_str" ]; then
    kill $VLLM_PID
    fi
}


task EvaluateIFEvalDA
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    < generations_path=$generations_path@GenerateIFEval
    > evaluations_path=evaluation.json
    :: evaluator_vllm_serve_str=@
    :: evaluator_completion_args=@
    :: evaluator_url=@
    :: evaluator=@
    :: vllm_env=@
    :: max_workers=@
{
    # init vllm if evaluator_vllm_serve_str is not empty
    wait_for_server() {
        local max_attempts=18000  # 5 hours timeout
        local attempt=0
        
        echo "Waiting for vLLM server to start..."
        while [ $attempt -lt $max_attempts ]; do
            if curl --silent --fail "${evaluator_url}/models" > /dev/null 2>&1; then
                echo "vLLM server is ready!"
                return 0
            fi
            
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "vLLM server process died unexpectedly"
                return 1
            fi
            
            ((attempt++))
            echo "Attempt $attempt/$max_attempts: vLLM server not ready yet, retrying in 1 second..."
            sleep 1
        done
        
        echo "Timeout waiting for vLLM server"
        return 1
    }
    if [ -n "$evaluator_vllm_serve_str" ]; then
        sleep 60 # wait a minute for vllm cleanup from previous tasks
        source $vllm_env/bin/activate
        vllm serve $evaluator_vllm_serve_str > vllm.log 2> vllm_error.log &
        VLLM_PID=$!
        if ! wait_for_server; then
            exit 1
        fi
    fi

    # run evaluations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python simple_evals.py \
        --eval=ifeval_da \
        --mode=evaluate \
        --grader-model=$evaluator \
        --grader-completion-args=$evaluator_completion_args \
        --n-threads=$max_workers \
        --generation-input=$generations_path \
        --output=$evaluations_path \
        --output-dir="$(dirname $evaluations_path)"
    
    # kill vllm server
    if [ -n "$evaluator_vllm_serve_str" ]; then
    kill $VLLM_PID
    fi
}

task EvaluateIFEvalPWC
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    < generations_path=$generations_path@GenerateIFEval
    < generations_b_path=$generations_b_path@GenerateIFEvalB
    > evaluations_path=evaluation.json
    :: evaluator_vllm_serve_str=@
    :: evaluator_completion_args=@
    :: evaluator_url=@
    :: evaluator=@
    :: vllm_env=@
    :: max_workers=@
{
    # init vllm if evaluator_vllm_serve_str is not empty
    wait_for_server() {
        local max_attempts=18000  # 5 hours timeout
        local attempt=0
        
        echo "Waiting for vLLM server to start..."
        while [ $attempt -lt $max_attempts ]; do
            if curl --silent --fail "${evaluator_url}/models" > /dev/null 2>&1; then
                echo "vLLM server is ready!"
                return 0
            fi
            
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "vLLM server process died unexpectedly"
                return 1
            fi
            
            ((attempt++))
            echo "Attempt $attempt/$max_attempts: vLLM server not ready yet, retrying in 1 second..."
            sleep 1
        done
        
        echo "Timeout waiting for vLLM server"
        return 1
    }
    if [ -n "$evaluator_vllm_serve_str" ]; then
        sleep 60 # wait a minute for vllm cleanup from previous tasks
        source $vllm_env/bin/activate
        vllm serve $evaluator_vllm_serve_str > vllm.log 2> vllm_error.log &
        VLLM_PID=$!
        if ! wait_for_server; then
            exit 1
        fi
    fi

    # run evaluations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python simple_evals.py \
        --eval=ifeval_pwc \
        --mode=evaluate \
        --grader-model=$evaluator \
        --grader-completion-args=$evaluator_completion_args \
        --n-threads=$max_workers \
        --generation-input=$generations_path \
        --generation-input-b=$generations_b_path \
        --output=$evaluations_path \
        --output-dir="$(dirname $evaluations_path)"
    
    # kill vllm server
    if [ -n "$evaluator_vllm_serve_str" ]; then
    kill $VLLM_PID
    fi
}



task EvaluateIFEvalTrue
    < ifeval_env=$ifeval_env@SetUpIFEvalEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    < generations_path=$generations_path@GenerateIFEval
    > eval_results_strict=eval_results_strict.jsonl
    > eval_results_loose=eval_results_loose.jsonl
{
    source $ifeval_env/bin/activate
    cd $evals_repo/instruction_following_eval

    python evaluation_main.py \
        --input_data=./data/input_data.jsonl \
        --input_response_data=$generations_path \
        --output_dir=$(dirname $eval_results_strict)
}


task MetaEvaluateHealthBench
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    > meta_evaluations_path=meta_evaluation.json
    > meta_evaluation_allresults=meta_evaluation_allresults.json
    :: evaluator_vllm_serve_str=@
    :: evaluator_completion_args=@
    :: evaluator_url=@
    :: evaluator=@
    :: vllm_env=@
    :: max_workers=@
{
    # init vllm if evaluator_vllm_serve_str is not empty
    wait_for_server() {
        local max_attempts=18000  # 5 hours timeout
        local attempt=0
        
        echo "Waiting for vLLM server to start..."
        while [ $attempt -lt $max_attempts ]; do
            if curl --silent --fail "${evaluator_url}/models" > /dev/null 2>&1; then
                echo "vLLM server is ready!"
                return 0
            fi
            
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "vLLM server process died unexpectedly"
                return 1
            fi
            
            ((attempt++))
            echo "Attempt $attempt/$max_attempts: vLLM server not ready yet, retrying in 1 second..."
            sleep 1
        done
        
        echo "Timeout waiting for vLLM server"
        return 1
    }
    if [ -n "$evaluator_vllm_serve_str" ]; then
        sleep 60 # wait a minute for vllm cleanup from previous tasks
        source $vllm_env/bin/activate
        vllm serve $evaluator_vllm_serve_str > vllm.log 2> vllm_error.log &
        VLLM_PID=$!
        if ! wait_for_server; then
            exit 1
        fi
    fi

    # run evaluations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python simple_evals.py \
        --eval=healthbench_meta \
        --grader-model=$evaluator \
        --grader-completion-args=$evaluator_completion_args \
        --n-threads=$max_workers \
        --output=$meta_evaluations_path \
        --output-dir="$(dirname $meta_evaluations_path)" \
        --mode=evaluate \
        --generation-input placeholder
    
    # kill vllm server
    if [ -n "$evaluator_vllm_serve_str" ]; then
    kill $VLLM_PID
    fi
}


task MetaEvaluateCommitteeHealthBench
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    < meta_evaluations_path=$meta_evaluations_path@MetaEvaluateHealthBench[Evaluator:*]
    < meta_evaluation_allresults=@MetaEvaluateHealthBench[Evaluator:*]
    > committee_meta_evaluations_path=meta_evaluation.json
    > committee_meta_evaluation_allresults=meta_evaluation_allresults.json
{
    # run evaluations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python healthbench_meta_eval_committee.py \
        --input-files $meta_evaluation_allresults \
        --output-dir="$(dirname $committee_meta_evaluations_path)"
}