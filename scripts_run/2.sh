# run an array of tasks sequentially
# array provided as variable
# provide how many completed or failed as well at end

#!/bin/bash
TASKS=(
    "python run.py ./configs/BTP2/bonn/bonn_moving_nonobstructing_box2.yaml | tee ./output/logs/BTP2/bonn_moving_nonobstructing_box2.log" 
    "python run.py ./configs/BTP2/bonn/bonn_person_tracking.yaml | tee ./output/logs/BTP2/bonn_person_tracking.log" 
    "python run.py ./configs/BTP2/viode/parking_lot_low.yaml | tee ./output/logs/BTP2/viode_parking_lot_low.log"
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