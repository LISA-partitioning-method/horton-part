#!/bin/bash

bash gen_api.sh
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --allow-errors *.ipynb
cd ../
make html  # Or sphinx-build command based on your setup
