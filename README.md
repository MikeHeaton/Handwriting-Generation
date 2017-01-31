# Handwriting-Generation

A Tensorflow implementation of Generating Sequences With Recurrent Neural Networks by Alex Graves,
applied to the problem of generating realistic handwriting.

Original paper: https://arxiv.org/pdf/1308.0850v5.pdf.</br>
Data source: http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database

Update 30/1/17: the basic generation network is working and trained. Sample weights are included in the /example_weights_stage1 folder. Here are some examples. Running generate.py will generate other examples:

![testplot_1](https://cloud.githubusercontent.com/assets/11911723/22485200/8c95be04-e7b9-11e6-957c-1070b900897f.png)
![testplot_2](https://cloud.githubusercontent.com/assets/11911723/22485245/c017a4fe-e7b9-11e6-8be3-a1ecbefd1f29.png)
![testplot_3](https://cloud.githubusercontent.com/assets/11911723/22485199/8c933c24-e7b9-11e6-87e8-622f4eb0ed4e.png)

At this checkpoint the network has been trained to generate handwriting based on previous inputs, but doesn't "know" which characters it's writing. Considering this, it's impressively legible! But the samples don't make grammatical sense, obviously. See Next Steps below.

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

### To Generate (out of the box):
* In config.py, point the weights_directory to "./example_weights_stage1/".
* Running generate.py will then generate a sample of text to testplot.png.

## Next steps
* I'm currently working on implementing part II of the methodology, feeding character information
to allow the network to learn what character it's writing. This will make the network able to write arbitrary text in realistic handwriting.
* Watch this space.
