# run an array of tasks sequentially
# array provided as variable
# provide how many completed or failed as well at end

#!/bin/bash
TASKS=(
    "python run.py ./configs/BTP2/advio_office/office_13.yaml |& tee ./output/logs/BTP2/advio_office_13.log" 
    "python run.py ./configs/BTP2/advio_office/office_14.yaml | tee ./output/logs/BTP2/advio_office_14.log" 
    "python run.py ./configs/BTP2/advio_office/office_15.yaml | tee ./output/logs/BTP2/advio_office_15.log"
    "python run.py ./configs/BTP2/advio_office/office_16.yaml | tee ./output/logs/BTP2/advio_office_16.log"
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