#!/bin/bash
# A shell script to deploy trained models to AWS Lambda

# Define AWS Lambda function name and model zip path
LAMBDA_FUNCTION_NAME="AnomalyDetectionFunction"
MODEL_PATH="path/to/your/model.zip"

# Deploy model to AWS Lambda
aws lambda update-function-code --function-name $LAMBDA_FUNCTION_NAME --zip-file fileb://$MODEL_PATH

# Output success message
echo "Model deployed to AWS Lambda successfully."
