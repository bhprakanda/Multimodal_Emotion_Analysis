#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=':4096:8'
export PYTHONHASHSEED='42'

# Remember to make the shell script executable:
# chmod +x setup_environment.sh