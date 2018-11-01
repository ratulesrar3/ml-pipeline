# Predicting Post-Release Interactions with Mental Health Systems
## stablegeniuses-mlpp2018 members:
* Hye Chang
* Ratul Esrar
* Sam Gallicchio
* Mario Moreno

## Project Summary
Our project attempted to build a model that can be used to identify people in the Johnson County jail system who are at risk of entering the county mental health system after their release from jail. Our goal was to identify inmates who would enter the county mental health system within one year of their release, based on information available to the jail system at the time of their release.

## Data
Our focus is on data held by Johnson County on inmates at the time of their release from jail. The outcome label depends on whether or not an individual goes on to access mental health services in the year after their release from jail.

## File Structure

    ├── pipeline		
    │   ├── setup.py              # database connection class
    │   ├── explore.py            # visualization helper functions
    │   ├── features.py           # python wrapper for sql feature generation
    │   ├── preprocess.py         # scripts for discretizing and cleaning data
    │   ├── methods_helper.py     # helper functions to run methods loop
    │   ├── methods_loop.py       # temporal validation loop functions
    │   └── evaluation.py         # takes results from validation loop to evaluate for bias and feature importance
    │      
    ├── documents
    │   ├── joco_paper.pdf
    │   └── joco_presentation.pdf
  	│  
    ├── Data_Exploration.ipynb    # notebook for exploratory data analysis           
  	│  
    ├── Models.ipynb	          # notebook to run pipeline
 	│
    ├── Evaluation.ipynb          # notebook for model evaluation
    │
    ├── config_example.json       # example db config file (private credentials should be stored locally)
    │  
    └── README.md

## Package Requirements
In a virtual environment:
```
pip install requirements.txt
```

## How to Replicate Our Results
1. Install required packages.
2. Setup connection to the proper database. Please refer to config_example.json to create your config.json file in the root directory.
3. Run `Data_Exploration.ipynb` notebook.
4. Pipeline:
* Run `Models.ipynb` notebook to loop over models-parameter combinations on your prediction window using temporal validation method.
* Model evaluation results and graphs are saved under */results* directory.
