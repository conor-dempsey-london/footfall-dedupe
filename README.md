# Intro
Small package for sampling from Bayesian models of footfall data. Initially for the purposes of providing a deduplication model which allows deduplicating footfall counts from BT data for a bespoke area, using model coefficients derived from a sample of areas where the total and deduplicated counts are known. 

# Setup

This project uses pixi for dependency management. Install pixi using the instructions [here](https://pixi.sh/latest/installation/).

Then clone the repo, enter the project directory and run
`
pixi install
`

This will install the dependencies in a virtual environment. 

Next create a .env file in the project folder and populate it with the following three variables:

DATA_BUCKET="-location of the S3 bucket containing your data files-"

COUNT_DATA_FILE="-your data file-.csv"

AREA_FILE="-file with areas of areas found in training data-.csv"

# Usage
Once you have set the environment variables to point to your training data, to sample from the models and output a lookup table of scale factors for each area (along with a lower and upper bound on these scale factors derived from the posterior distributions over the model parameters), run the following:

`
pixi run python ./fit_models_output_lookup.py
`