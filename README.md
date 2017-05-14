# Process Sequence

### Directory structure:

```
project
│   README.md
│   requirement.txt    
│   report.pptx
|
└───code: code of reference 
│   │   train.py
|   |   evaluate_next_activity_and_time.py
│   │   evaluate_suffix_and_remaining_time.py
│   │   calculate_accuracy_on_next_event.py
|   |
│   └───output_files
| 
└───data
|   │   helpdesk.csv: full data for analysis
|
└───experiment
|   |   create_df.ipynb
|   |   prepare_data.ipynb
|
└───input
|
└───reference
```
### Reference

The code of [reference model](https://github.com/verenich/ProcessSequencePrediction) was implemented in Python 2.7. To run:

1. Install requirement

```pip install -r requirements.txt```

2. Create dirs for output:

```mkdir code/output_files/folds```

```mkdir code/output_files/models```

```mkdir code/output_files/results```

3. Run ```python train.py```
4. Run ```python evaluate_next_activity_and_time.py``` and ```python evaluate_suffix_and_remaining_time.py```
5. Run ```python calculate_accuracy_on_next_event.py```
