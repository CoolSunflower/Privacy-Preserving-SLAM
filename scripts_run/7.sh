# run an array of tasks sequentially
# array provided as variable
# provide how many completed or failed as well at end

#!/bin/bash
TASKS=(
    "python run.py ./configs/BTP1/tum_rgbd/f3_sr.yaml | tee ./output/logs/BTP1/tum_f3_sr.log" 
    "python run.py ./configs/BTP1/tum_rgbd/f3_ws.yaml | tee ./output/logs/BTP1/tum_f3_ws.log" 
    "python run.py ./configs/BTP1/tum_rgbd/f3_wx.yaml | tee ./output/logs/BTP1/tum_f3_wx.log"
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