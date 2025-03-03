# PHBO 

## Installation
Install all required packages using  
`pip install -r requirements.txt`

## Directory Structure
- **/data_record**  
  Contains data reading/output modules and case data files

- **/PHBO**  
  Contains code necessary for computations

## Usage Instructions
1. Execute the `main.py` file to read experimental data and provide recommended points

2. **Current Configuration**  
   - Acquisition function in use **pr_ei**  
   - For real-world experimental systems **enzyme** 
     The program will automatically stop after completing one batch iteration and output recommended point data
