# This is an automated calculation script for the acc, acc+, and score.

# You can directly run "python3 calculation.py" to get the evaluation results of LaVIN.

# In order to get the statistical results of your model:

(1) Fill all the files in "Your_Results", adding your model's responses:
Each file in "Your_Results" consists of:
Image_Name + "\\t" + Question + "\\t" + Ground_Truth_Answer + "\\n"

You need to add the responses of your model as:
Image_Name + "\\t" + Question + "\\t" + Ground_Truth_Answer + "\\t" + Your_Response + "\\n"

Note: if your responses contain "\\n", please delet it. For each question, your response can only be in one line, not across lines!

(2) run "python3 calculation.py --results_dir ./Your_Results"
