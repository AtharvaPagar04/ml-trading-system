#!/bin/bash

echo "Starting sequential multi-seed training..."

SEEDS=(42 111 555 999)

for SEED_VAL in "${SEEDS[@]}"
do
  echo "------------------------------------"
  echo "Running training with SEED=$SEED_VAL"
  echo "------------------------------------"

  SEED=$SEED_VAL python pattern_model.py | tee run_${SEED_VAL}.log

  echo "Completed SEED=$SEED_VAL"
done

echo "All runs finished."

