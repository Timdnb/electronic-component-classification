Notes from http://jvgemert.github.io/paper_skeleton.rtf:

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
- Not clear what experimental questions you ask that demonstrate to solve problem and consequences

WHY INTERESTING: What is the general setting/application. Why should people care.
- The user will be able to take a photo of a sketched electronic circuit and pass it through a program to locate the components and junctions
- These components and junctions can then be replaced by neat digital versions
- Ultimately this will allow translation of sketched electronic circuits to prototypes in a digital form

HOW DONE NOW: The typical approach(es) to the setting in (1)
- Currently there is no way to digitize electronic circuit sketches
- Some models exist that have been trained to detect the components of a hand-drawn circuit
- A subset of these also detect line junctions

WHAT IS MISSING: What’s the problem in (2), and what consequences does this have.
- Lines are currently NOT being detected
- There is NO knowlege about how components are connected to each other
- The current classifiers merely classify the components and junctions
- This means the circuit needs to be digitally created by hand anyways

PROPOSED SOLUTION: What do you do, and why does it solve the problem in (3)
- To solve the problem we will create custom datasets for components and junctions
- We will train eiter one classifier using YOLO to detect both components and junctions or two classifiers, one for components and one for junctions in series
- We will also try to draw connection lines between the components and junctions.

EXPERIMENTAL QUESTIONS: How do you evaluate experimentally that (4) solves the problem in (2) and it’s consequences in (3).
- The solution is succesful if it can detect and classify all components and junctions
- Additionally the components and junctions are replaced by digital counterparts
- If time and resources allow, the components are connected with lines digitally too
- Creating neat and complete digital versions of electronic ciruict sketches

### Note from Tim
almost done imo, cleaned it up a bit and added a few things. the only thing we might want to change is better emphasize that our first goal is to classify components and junctions and that if we have time we will also create a digital version
