# run an array of tasks sequentially
# array provided as variable
# provide how many completed or failed as well at end

#!/bin/bash
TASKS=(
    "python run.py ./configs/BTP1/tum_rgbd/f3_ss.yaml | tee ./output/logs/BTP1/tum_f3_ss.log" 
    "python run.py ./configs/BTP1/tum_rgbd/f3_sx.yaml | tee ./output/logs/BTP1/tum_f3_sx.log" 
    "python run.py ./configs/BTP1/tum_rgbd/f3_shs.yaml | tee ./output/logs/BTP1/tum_f3_shs.log"
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