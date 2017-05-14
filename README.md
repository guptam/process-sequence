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
|   │   helpdesk_copy.csv: small data for checking
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

To run the code of [reference model](https://github.com/verenich/ProcessSequencePrediction):
1. Install requirement

```pip install -r requirements.txt```

2. Create dirs for output:

```mkdir code/output_files/folds```

```mkdir code/output_files/models```

```mkdir code/output_files/results```

3. Run ```train.py```
4. Run ```evaluate_next_activity_and_time.py``` and ```evaluate_suffix_and_remaining_time.py```
5. Run ```calculate_accuracy_on_next_event.py```
