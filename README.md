# PHBO

## Installation

Install all required packages using:

```
pip install -r requirements.txt
```

Estimated installation time: **5 minutes**

---

## Directory Structure

- **/data_record**  
  Contains modules for data reading and output, along with case-specific data files.

- **/PHBO**  
  Contains core modules for computation and Bayesian optimization logic.

---

## Usage Instructions

1. Run the `main.py` file to read existing experimental data and generate new recommended points.

2. **Current Configuration**  
   - **Acquisition Function**: `pr_ei`  
   - **Application Scenario**: `enzyme` (real-world experimental system)  
   The program will **automatically stop** after completing one batch of optimization and will output the recommended experimental points.

---

## Example Run

To execute the example workflow:

```
python main.py
```

This will read the existing experimental data from `case.xlsx` and generate recommended point data using default parameters. The output will be stored in the `enzyme` folder.

Estimated runtime: **3 minutes**

---

