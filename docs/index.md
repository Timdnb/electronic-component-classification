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

## The idea
introduction / explain possible usecase / to learn and for fun

with this blogpost we hope to inspire other deep learning projects...

## The data
explain general data necessary and maybe provide some sources, explain why we decided to create own datasets

### Components dataset
explain data(set), why this data, link to dataset?
labeling procedure
show sample + labels
explain which notebook used for reference

### Junctions dataset
explain data(sets), why this data, link to dataset?
labeling procedure
show sample + labels
explain which notebook used for reference

## Training
which model, epochs, rotations (in)variance, other considerations

## Pipeline explanation
image -> data preprocessing -> through model 1 -> through model 2 -> data post processing -> labeled image (for now, ideally digital version)

explain why this pipeline and other considerations that we had (e.g. that we first wanted to delete components and then detect junctions)

upload models to huggingface (or similar) and add links

for testing and investigation (of code) can reference to inference.ipynb, however in the end we should make a .py file that does everything

## Results
couple sample images with results
- one complicated circuit
- sheet with one of every component?

then couple examples of poor performance
- effect of poor preprocessing (show multiple side to side with diff preprocessing)
- effect when circuit does not adhere to standards (90 deg turns)

metrics
- model performance on test set?

## Discussion / future work
what works, what doesnt, what would be the next step, how can it be improved

## Closing
Encourage to build on this work, all code is open source and so are the datasets (need to check if I can make components dataset public, maybe it's copyright lol) and models

Thank TA

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