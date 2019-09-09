# CS7642_Project2

Dear Instructors and TAs,

Thanks for your time to review my codes and project. Here are few explainations about the folders and codes in this repository.

1. Codes for training and testing models:

./learningRate

Codes for tuning up learning rates.

./discount

Codes for tuning up discount factors.

./epsilon

Code for performing linear epsilon-decay strategy.

./logcosh

Code for using logcosh as loss function.

./final_model

Code for training and testing final model

For codes in the above five folders, just simply type the following command in command line:

$> python FILE_NAME.py

2. Codes for plotting:

./plot

Code for plotting the figures in project report, and it need specify where the output files are(OUTPUT_FILE_DIR). These files were generated during training and testing models:

$> python project2_plot.py OUTPUT_FILE_DIR

For instance:

$> python project2_plot.py ./output

3. Output files generated during training and testing models:

These files are saved under ./output folder, and could be used to plot figures in project report along with ./plot/project2_plot.py.

All codes were compiled using python (3.4.5 :: Anaconda custom (64-bit)) and keras (2.1.5) with Theano (1.0.1) as the backend. Since I setup the seed in each file, you could revisit my results. Let me know if you have any further concerns.

Best,

Zheng Fu / zfu66@gatech.edu
