#!/bin/bash

# Create a virtual environment
python3 -m venv envVis4

# Activate the virtual environment
source envVis3/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate