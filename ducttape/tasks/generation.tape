import setup.tape

task GenerateHealthBench
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    > generations_path=generation.jsonl
    :: generator_vllm_serve_str=@
    :: generator_completion_args=@
    :: generator_url=@
    :: generator=@
    :: vllm_env=@
    :: max_workers=@
{
    # init vllm if generator_vllm_serve_str is not empty
    wait_for_server() {
        local max_attempts=18000  # 5 hours timeout
        local attempt=0
        
        echo "Waiting for vLLM server to start..."
        while [ $attempt -lt $max_attempts ]; do
            if curl --silent --fail "${generator_url}/models" > /dev/null 2>&1; then
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
    if [ -n "$generator_vllm_serve_str" ]; then
        sleep 60
        source $vllm_env/bin/activate
        vllm serve $generator_vllm_serve_str > vllm.log 2> vllm_error.log &
        VLLM_PID=$!
        if ! wait_for_server; then
            exit 1
        fi
    fi

    # run generations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python simple_evals.py \
        --eval=healthbench \
        --mode=generate \
        --model=$generator \
        --model-completion-args=$generator_completion_args \
        --n-threads=$max_workers \
        --output=$generations_path
    
    # kill vllm server
    if [ -n "$generator_vllm_serve_str" ]; then
    kill $VLLM_PID
    fi
}

task GenerateIFEval
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    > generations_path=generation.jsonl
    :: generator_vllm_serve_str=@
    :: generator_completion_args=@
    :: generator_url=@
    :: generator=@
    :: vllm_env=@
    :: max_workers=@
{
    # init vllm if generator_vllm_serve_str is not empty
    wait_for_server() {
        local max_attempts=18000  # 5 hours timeout
        local attempt=0
        
        echo "Waiting for vLLM server to start..."
        while [ $attempt -lt $max_attempts ]; do
            if curl --silent --fail "${generator_url}/models" > /dev/null 2>&1; then
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
    if [ -n "$generator_vllm_serve_str" ]; then
        sleep 60
        source $vllm_env/bin/activate
        vllm serve $generator_vllm_serve_str > vllm.log 2> vllm_error.log &
        VLLM_PID=$!
        if ! wait_for_server; then
            exit 1
        fi
    fi

    # run generations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python simple_evals.py \
        --eval=ifeval \
        --mode=generate \
        --model=$generator \
        --model-completion-args=$generator_completion_args \
        --n-threads=$max_workers \
        --output=$generations_path
    
    # kill vllm server
    if [ -n "$generator_vllm_serve_str" ]; then
    kill $VLLM_PID
    fi
}



task GenerateIFEvalB
    < healthbench_env=$healthbench_env@SetUpHealthBenchEnv
    < evals_repo=$simple_evals_repo@CloneSimpleEvals
    > generations_b_path=generation.jsonl
    :: generatorb_vllm_serve_str=@
    :: generatorb_completion_args=@
    :: generatorb_url=@
    :: generatorb=@
    :: vllm_env=@
    :: max_workers=@
{
    # init vllm if generator_vllm_serve_str is not empty
    wait_for_server() {
        local max_attempts=18000  # 5 hours timeout
        local attempt=0
        
        echo "Waiting for vLLM server to start..."
        while [ $attempt -lt $max_attempts ]; do
            if curl --silent --fail "${generator_url}/models" > /dev/null 2>&1; then
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
    if [ -n "$generator_vllm_serve_str" ]; then
        sleep 60
        source $vllm_env/bin/activate
        vllm serve $generator_vllm_serve_str > vllm.log 2> vllm_error.log &
        VLLM_PID=$!
        if ! wait_for_server; then
            exit 1
        fi
    fi

    # run generations
    source $healthbench_env/bin/activate
    cd $evals_repo
    python simple_evals.py \
        --eval=ifeval \
        --mode=generate \
        --model=$generator \
        --model-completion-args=$generator_completion_args \
        --n-threads=$max_workers \
        --output=$generations_path
    
    # kill vllm server
    if [ -n "$generator_vllm_serve_str" ]; then
    kill $VLLM_PID
    fi
}