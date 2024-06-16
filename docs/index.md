---
layout: default
---

<!-- PENALTIES FOR BLOG
- No results, or not enough motivation for why there are no results
- Not enough effort shown
- Results are not explained
- Results are inconsistent and not motivated
- Insufficient "computer vision" alignment
- Unclear why an experiment is done (what question is answered by it, and why is this question interesting)

- The text is not stand-alone; it's not peer understandable.
- Using a term before defining/motivating it.
- Unclear logical reasoning step.
- Inconsistent use of terminology.
- Too much unnecessary detail. -->

**By:** Tim den Blanken, Felipe Bononi Bello, Miquel Rull Trinidad

**Course:** CS4245 - Seminar Computer Vision by Deep Learning

# The idea
Sketching is one of the most effective ways to communicate ideas. It is a common practice in engineering, where engineers use sketches to communicate their ideas to other engineers, clients, or even to themselves. However, the process of converting these sketches into digital formats is time-consuming and error-prone. In this project, we aim to automate this process by developing a deep learning model that can detect and classify components and junctions in sketches of electronic circuits. 

Apart from the practical applications of this project, it also serves as a learning experience for us, and possibly for the reader. This blog post will document our journey from the idea above to the final product. We go over the challenger of working with real-world data, the importance of preprocessing and other intricacies of training deep learning models. We also aim to have fun while working on this project, as we are all passionate about computer vision and deep learning.

# The data
Since we are looking to classify hand-drawn sketches of electronic circuits, we need a dataset that represents such skethes. However clean and well-annotated datasets for this task are hard to find. We did however find a dataset of hand-drawn electronic components [^1], which are obviously the building blocks of electronic circuits. There is still a difference between classifying a single components or detecting them as part of a circuit. For this reason we decided to create our own dataset using hand-drawn circuits. Since we also want to understand the connections between components, we have also created a dataset that can be used to detect junctions in circuits. Both datasets will be explained in more detail in the next sections.

## Components dataset
As mentioned above, we found a dataset of hand-drawn electronic components. This dataset contains 15 classes of components of the arguably the most common components in electronic circuits. The dataset contains about 200 images per class. The next step was to use those to create a dataset that can be used to train a YOLO model, as we want to detect the components. To do this we generate images with the components randomly scattered across. To improve performance we add random lines and shapes to confuse the model, and we apply random noise. The labels are created from the original image of the separate components, however since those were always square with the components not covering the entire image, we had to adjust the labels to fit the new images. This was simply done by finding the edges of the components and adjusting the labels accordingly. After these steps a training sample looks as follows:

![Components dataset sample](https://i.imgur.com/HtsrY87.png)

The red boxes show the bounding boxes. As you see the bounding boxes are pretty good, but sometimes they are not perfect. This is because the labels were created automatically to save time. Regardless, the model will be able to learn from this data. The dataset can be found [here](https://www.kaggle.com/datasets/timdnb/components). The notebook that was used to train the model can be found in the repository in the notebooks folder as `component_dataset_generation.ipynb`

## Junctions dataset
The junctions dataset uses a different approach. Due to the lack of available datasets, it was decided to generate our own dataset. To keep it simple we decided to only use junctions with 90 degree angles. The process can be followed in `juntion_synthetic_generator.ipynb` in the notebooks folder.

Given the 90-degree angled junctions, there are 9 types (four corners, four 3-way junctions and one 4-way junction). These junctions are labeled with a four digit number of ones and zeros. A digit is a 1 if there is a line in the corresponding part of the junction and 0 otherwise. The order of the digits is: down, up, left, right. Therefore if a label is junction0110, it means the junction joins up and left. To generate each junction, openCV lines were generated according to the labels, with some randomisation on the angles. This is to imitate the human drawings, as most junctions will not be perfect 90 degree angles. After generating the lines, gaussian blur and noise is added. There is further randomisation by making the lines different thicknesses and by varying the size of the junctions from 100 to 300 pixels. Each junction has been generated 1000 times, for a total of 9000 juntion images. Some examples of junctions can be seen below.

![Generated junctions](https://i.imgur.com/dg05O4t.jpg)

After generating the junctions, a dataset generator similar to the one for the dataset was used. The only difference being that the extra random lines and shapes were removed, as they can sometimes resemble junctions. Finally, only the center portion of the junctions is labeled, such that the line crosses the bounding box. This ensures that the model will detect junctions that are part of a circuit, and not in isolation. In total, 2402 images were generated for training and 503 for validation. The dataset can be found [here](https://www.kaggle.com/datasets/miquelrulltrinidad/junctions).

![Junctions train image](https://imgur.com/i49kIoc.jpg)
<!-- possible TODO: add bounding box labels -->

# Training
TODO: Explain metrics (mAP50 and 95)  
For the training of the models we have chosen to use [YOLOv5m](https://github.com/ultralytics/yolov5) for its simplicity and well-proven performance. Two models were trained separately utilizing 2 T4 or a single P100 GPUs  in Kaggle.  

The model to detect components was trained with the standard hyperparameters at an image size of 640x640 and batch size of 32. The model has been trained for 25 epochs, leading to the following performance on the validation set:

![Components model performance](https://imgur.com/gY9yswO.jpeg)

The model to detect the junctions was trained with the standard hyperparameters at an image size of 640x640 and batch size of 42. The model has been trained for 20 epochs, leading to the following performance on the validation set:

![Junctions model performance](https://imgur.com/bEfHqq4.jpeg)

# Pipeline explanation
<!-- image -> data preprocessing -> through model 1 -> through model 2 -> data post processing -> labeled image (for now, ideally digital version)

explain why this pipeline and other considerations that we had (e.g. that we first wanted to delete components and then detect junctions, also sliders for preprocessing came later since it was hard to have one set of values that just works)

upload models to huggingface (or similar) and add links
https://huggingface.co/Timdb/electronic-circuit-detection/tree/main
^^ contains both models

for testing and investigation (of code) can reference to inference.ipynb, however in the end we should make a .py file that does everything -->

In our project, we developed a specialized machine learning pipeline to enhance the accuracy of detecting and classifying components and junctions in electronic circuit sketches. The pipeline has undergone several iterations to optimize its functionality and address various challenges encountered along the way. The final pipeline consists of the following blocks:

<!-- During the duration of the project the pipeline has been expanded and changed to best fit the goal. The first iteration only made use of a component detection model, after which we thought to add junction labelling capability to the model. This however did not work as expected, as some of the components have junction-like parts to them which causes confusion. So to be able to fulfill the goal of detecting and classifying components and junctions in sketches of electronic circuits, we have created the following pipeline that includes preprocessing, two detection models and postprocessing: -->

![Model Pipeline](https://imgur.com/ZnRoW7G.png)

1. **Data Preprocessing:** The initial sketch, ideally drawn in black or blue on a white page without background lines, is converted to a greyscale image. This image is then inverted to a black background with white lines, where pixel values above a certain threshold are turned black and those below are turned white. This threshold can be adjusted to suit different lighting conditions, enhancing the flexability of our preprocessing stage.

2. **Component Detection Model:** The first model in our pipeline processes the preprocessed image to identify and locate electronic components. It outputs the position, size and detection probability of each component found in the image.

3. **Junction Detection Model:** In parallel, the same preprocessed image is fed into our second model designed to detect junctions. This model outputs the positions, sizes, and probabilities of any detected junctions.

4. **Data Postprocessing:** Given that some components posses junction-like features, both models occassionaly recognize the same feature, leading to overlapping detections. Our postprocessing stage addresses this by suppressing junction detections that occur within the boundaries of detected components, ensuring clear and distinct output labels.


Initially, our pipeline only included a component detection model. However, to enhance its capabilities, we integrated a junction detection model. Adjusting to the challenge that some components mimic the appearance of junctions, we refined our models and added a postprocessing step to resolve these ambiguities.

We also introduced adjustable sliders for preprocessing to handle diverse image conditions effectively, a feature added after recognizing the limitations of a one-size-fits-all threshold value.

The final models are hosted on Hugging Face and can be accessed [here](https://huggingface.co/Timdb/electronic-circuit-detection/tree/main). These models can also be used to run the `inference.ipynb` notebook, which goes over the steps individually.
# Results

In order to be able to showcase the performance of the created models we have created handdrawn circuits of which a few are portrayed in this section. A sheet with all individual electronical components, an AC-to-DC converter and some example circuits showing examples of poor performance due to preprocessing mistakes.

<div style="display: flex; flex-wrap: wrap; justify-content: space-around; align-items: stretch;">
  <div style="width: 45%; margin: 5px;">
    <img src="https://imgur.com/nqDnhg5.jpeg" alt="Electrical components sheet" style="width: 100%; height: 100%; object-fit: cover;">
  </div>
  <div style="width: 45%; margin: 5px;">
    <img src="https://imgur.com/bipLNRV.png" alt="AC-to-DC converter" style="width: 100%; height: 100%; object-fit: cover;">
  </div>
</div>

Now we will show a few examples of bad performance due to preprocessing mistakes. In the first image, the contrast value in the preprocessing is not set high enough, which causes the component models to hallucinate and 'detect' two components in the bottom left corner.

<div style="display: flex; flex-wrap: wrap; justify-content: space-around; align-items: stretch;">
  <div style="width: 45%; margin: 5px;">
    <img src="https://imgur.com/jjpxZxn.png" alt="Model hallucinations" style="width: 100%; height: 100%; object-fit: cover;">
  </div>
  <div style="width: 45%; margin: 5px;">
    <img src="https://imgur.com/oF8xq6E.png" alt="Too low contrast value" style="width: 100%; height: 100%; object-fit: cover;">
  </div>
</div>

The second example of poor performance is due to the circuit not adhering to the standards and being tilted by 90 degrees. Where one can see that the ground component, the ammeter, and the battery are not detected correctly. When the ammeter is tilted by 90 degrees it is often mistaken for the dc_volt_src or the curr_src.

<div style="display: flex; flex-wrap: wrap; justify-content: space-around; align-items: stretch;">
  <div style="width: 45%; margin: 5px;">
    <img src="https://imgur.com/k17N1ju.png" alt="Tilted circuit" style="width: 100%; height: 100%; object-fit: cover;">
  </div>
  <div style="width: 45%; margin: 5px;">
    <img src="https://imgur.com/HcldKjL.png" alt="Non-tilted circuit" style="width: 100%; height: 100%; object-fit: cover;">
  </div>
</div>


<!-- couple sample images with results
- one complicated circuit
- sheet with one of every component

then couple examples of poor performance
- effect of poor preprocessing (show multiple side to side with diff preprocessing)
- effect when circuit does not adhere to standards (90 deg turns)

metrics
- model performance on test set? -->

# Discussion / future work
The results have shown promising performance in some cases, but a few issues are apparent too. One of the main problems comes from the preprocessing step, as this is a manual step where it sometimes is near impossible to extract the circuit properly. Therefore in the current state it is recommended to use full white paper and write with a dark colored pen.

A second shortcoming follows from the detection of components. The model tends to have some trouble detecting the 'dc_volt_src_2', which is a volt source (small vertical line followed by empty space and then a long vertical line). The other components it is detecting quite consistently. 

Another limitation regarding the component detection is that not all components can be detected under any rotation. More specifically, only the capacitors, diodes, inductors and resistors can have any rotation. The other components should be upright with respect to the camera direction.

The junction detection model performs very well and rarely makes mistakes. However currently it can only detect junctions with 90 degree angles. This means that any circuit that does not adhere to this will not be completely labelled as intended.

This brings us to the future improvements that directly follow from the points above:
1. More robust and automatic extraction of circuit from image
2. Better component detection model, able to consistently detect all components under any orientation
3. Better junction detection model, able to detect any angle junction

Apart from this, there are more parts that the program can improve on:

4. Possibly a version of YOLO that has oriented bounding boxes [^2] can be used, this can lead to a more robust algorithm, that can have components under any angle, and not just 90 degree angles 
5. The components model can detect the most used components, but not all possible components are included. This could be expanded.
6. Currently the program only detects the components and junctions and returns a labelled image. The next step would be to take the labels and actually digitize the sketch of the electronic circuit.

<!-- what works, what doesnt, what would be the next step, how can it be improved
- want a fully digitized version -->

# Closing
We encourage the reader to build on this work. All code is open-source and so are the datasets and models. We hope that this project can inspire other deep learning projects and that it can be used as a learning resource for those interested in computer vision and deep learning.

Lastly, we would like to thank our supervisor Xiangwei Shi for his guidance and support throughout this project.

# References
[^1]: https://www.kaggle.com/datasets/moodrammer/handdrawn-circuit-schematic-components
[^2]: https://docs.ultralytics.com/tasks/obb/
 
<!-- CAN USE STUFF BELOW TO LOOK UP HOW TO CREATE CERTAIN STYLES OF TEXT -->

<!-- Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
``` -->