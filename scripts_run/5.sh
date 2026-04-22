# run an array of tasks sequentially
# array provided as variable
# provide how many completed or failed as well at end

#!/bin/bash
TASKS=(
    "python run.py ./configs/BTP2/tum_rgbd/f3_wx.yaml | tee ./output/logs/BTP2/tum_f3_wx.log" 
    "python run.py ./configs/BTP2/tum_rgbd/f3_whs.yaml | tee ./output/logs/BTP2/tum_f3_whs.log" 
    "python run.py ./configs/BTP2/tum_rgbd/f3_wr.yaml | tee ./output/logs/BTP2/tum_f3_wr.log"
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