# run an array of tasks sequentially
# array provided as variable
# provide how many completed or failed as well at end

#!/bin/bash
TASKS=(
    "python run.py ./configs/BTP1/bonn/bonn_balloon2.yaml | tee ./output/logs/BTP1/bonn_balloon2.log" 
    "python run.py ./configs/BTP1/bonn/bonn_crowd.yaml | tee ./output/logs/BTP1/bonn_crowd.log" 
    "python run.py ./configs/BTP1/bonn/bonn_crowd2.yaml | tee ./output/logs/BTP1/bonn_crowd2.log"
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