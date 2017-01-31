# Handwriting-Generation

A Tensorflow implementation of Generating Sequences With Recurrent Neural Networks by Alex Graves,
applied to the problem of generating realistic handwriting.

Original paper: https://arxiv.org/pdf/1308.0850v5.pdf.
Data source: http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database

Update 30/1/17: the basic generation network is working and trained. Sample weights are included in the /network_weights folder. Here are some examples, running generate.py will generate other examples:

![testplot_1](https://cloud.githubusercontent.com/assets/11911723/22485200/8c95be04-e7b9-11e6-957c-1070b900897f.png)
![testplot_2](https://cloud.githubusercontent.com/assets/11911723/22485245/c017a4fe-e7b9-11e6-8be3-a1ecbefd1f29.png)
![testplot_3](https://cloud.githubusercontent.com/assets/11911723/22485199/8c933c24-e7b9-11e6-87e8-622f4eb0ed4e.png)

## Instructions for use

### To Train:
* Download the data from data source link above (you will need to register).
* Unzip the data to a directory inside the project, and point config to the directory
containing the samples (and nothing else).
* Run preprocess_data.py; this will calculate the mean and variance of the data in your
samples directory, so that sample offsets can be normalised when training and generating.
* Edit parameters in config.py as desired - as of 31/1/17, defaults should work.
* Run train.py to train. Training for stage 1 (without character differentiation) took ~18 hours on my
Macbook local.

### To Generate:
* In config.py, point the weights_directory to "./example_weights_stage1/".
* Running generate.py should then generate a sample of text to testplot.png.
