# run an array of tasks sequentially
# array provided as variable
# provide how many completed or failed as well at end

#!/bin/bash
TASKS=(
    "python run.py ./configs/BTP2/tum_rgbd/f3_shs.yaml | tee ./output/logs/BTP2/tum_f3_shs.log" 
    "python run.py ./configs/BTP2/tum_rgbd/f3_sr.yaml | tee ./output/logs/BTP2/tum_f3_sr.log" 
    "python run.py ./configs/BTP2/tum_rgbd/f3_ws.yaml | tee ./output/logs/BTP2/tum_f3_ws.log"
)

COMPLETED=0
FAILED=0

for TASK in "${TASKS[@]}"; do
    echo "Running: $TASK"
    eval $TASK
    if [ $? -eq 0 ]; then
        echo "Completed: $TASK"
        ((COMPLETED++))
    else
        echo "Failed: $TASK"
        ((FAILED++))
    fi
done

echo "Summary:"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"