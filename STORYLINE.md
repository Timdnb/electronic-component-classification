<!-- Notes from http://jvgemert.github.io/paper_skeleton.rtf:

The story line is one of the most important tools in research: It helps to structure thoughts, 
meetings, the process, the research questions, and enormously helps the writing. It is normal 
that the story line changes/evolves many times, see: http://jvgemert.github.io/HowToDoResearchInDL-slides.pdf 

Each element might take 1-3 bullet points.
Important: The story line is stand alone: you cannot introduce a new term/concept before it 
has been introduced (by motivating why the term/concept is important).

Stand alone high level story line (1 bullet per topic; before using a new term/concept make 
sure to first motivate the term/concept (ie: what is it and why is it needed). Terms/concepts 
logically build/connect to earlier terms/concepts (ie: its a story). Have short “1-liners” per 
bullet point; correct grammar is optional; the whole story line should fit on half a page.)

Penalties for storyline:
- Not clear why interesting or why people should care.
- Not clear how it's done now: what current typical approach(es).
- Not clear what is missing; what the problem is, and what consequences this problem has.
- Not clear what you propose; what your method is/does. 
- Not clear why your proposal/method could solve the problem.
- Not clear what experimental questions you ask that demonstrate to solve problem and consequences -->

# Storyline

1. WHY INTERESTING: What is the general setting/application. Why should people care.
- It is useful to be able to recognise hand drawn circuits during prototyping and for education purposes
- The drawn components and junctions can then be replaced by neat digital versions. Ultimately this will allow translation of sketched electronic circuits to digital form

2. HOW DONE NOW: The typical approach(es) to the setting in (1)
- Currently there is no way to digitize electronic circuit sketches
- Some models exist that have been trained to detect the components of hand-drawn circuits
- A subset of these also detect line junctions

3. WHAT IS MISSING: What’s the problem in (2), and what consequences does this have.
- Current classifiers merely classify components and are not ready to be transformed into digital images

4. PROPOSED SOLUTION: What do you do, and why does it solve the problem in (3)
- To solve the problem we will create custom datasets for components and junctions. The junctions will have orientation encoded in the labels to facilitate future digitalisation
- We will train two classifiers, one for components and one for junctions

5. EXPERIMENTAL QUESTIONS: How do you evaluate experimentally that (4) solves the problem in (2) and it’s consequences in (3).
- What is the performance regarding detection and classificatoin of components and junctions in terms of accuracy and IoU on an individual basis?
- How do both detectors perform when working in series in a pipeline?