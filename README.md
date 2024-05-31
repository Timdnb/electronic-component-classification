# Electronic Circuit Detection
As part of the course CS4245 Seminars Computer Vision by Deep Learning at the Delft University of Technology, we have developed a pipeline in order to convert sketches of electronic circuit into a labeled image. For a detailed overview of this project, please visit the [blogpost](https://timdnb.github.io/electronic-component-classification/).

## Installation
First create a virtual environment and install dependencies (tested for Python 3.10.12)
```
git clone git@github.com:Timdnb/electronic-component-classification.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
Take an image of your electronic circuit and store it, then run the following command

```
python detect.py --image /path/to/image [--outdir /path/to/outdir]
```
Arguments:
- `--image`: path to your image file
- `--outdir`: path where you want to output to be stored, this is an optional parameter

After running a window will appear, in this window use the sliders to best extract the lines of your electronic circuit. Once you are satisfied press "Enter" to continue.

## Inference example
In the current state the pipeline classifies all components and junctions, however it does not yet convert it to a digital counterpart. This can be a future improvement

Original:

<img src="assets/example_image.jpg" alt="electronic circuit" width="500"/>

Labeled:

<img src="assets/example_image_output.jpg" alt="labeled electronic circuit" width="500"/>

## Additional information
The `notebooks` folder contains all notebooks that were used during development, they can be investigated to have a better look into our methods. However one of the notebooks requires the presence of a hand-drawn electronic component dataset from [kaggle](https://www.kaggle.com/datasets/moodrammer/handdrawn-circuit-schematic-components). Download this dataset and put the folders of the separate components under a `dataset/components` directory (make this directory yourself).