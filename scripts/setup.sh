#!/bin/bash

# Setup Backend
echo "Setting up Backend..."
cd backend
if command -v conda &> /dev/null; then
    conda env update --file environment.yml --prune
    # Activate handled by user or subshell usually, but we assume environment is ready
else
    pip install -r requirements.txt
fi
cd ..

# Setup Frontend
echo "Setting up Frontend..."
cd frontend
npm install
cd ..

echo "Setup Complete."
