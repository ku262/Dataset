# EndoGaze

The available code here can be used to obtain validation results in the article.

### Evaluation folder
The Evaluation folder contains validation related scripts. Blink.py is used to calculate the number of blinks, detection.py is used to calculate the detection rate, eye_travel. py is used to calculate eye movement distance, fixation. py is used to calculate the proportion of gaze points in each area, heatmap. py is used to calculate heatmaps, reaction. py is used to calculate reaction time and generate relevant CSV files. utils.py includes some tools.
### Main Program
As the main program, you must modify the address from lines 54 to 56 to your address, and get_file will generate a CSV file for calculating the detection rate and reaction time. The main will calculate the detection rate, eye movement distance, distribution ratio of gaze points in different regions, blink frequency, and optional heatmaps. The relevant data will be generated into a CSV file and saved in the output path.
### Box plot
If you want to draw a box plot of the gaze point, you need to replace lines 62-63 with your address, and executing the handle will generate the box plot.
