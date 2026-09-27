# Cognitive Visual Reasoning Agent for Raven's Progressive Matrices

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-image%20I%2FO-blue)
![Course](https://img.shields.io/badge/Georgia%20Tech-CS%207637%20Knowledge--Based%20AI-B3A369)
![No neural network](https://img.shields.io/badge/learning-none%20(symbolic%20%2B%20pixel%20heuristics)-C9A25A)

A Python agent that solves Raven's Progressive Matrices, the abstract visual reasoning puzzles used in psychometric testing, from raw images alone. No training data, no neural network. The agent perceives each puzzle with OpenCV, reasons about the relationships between cells using pixel-level and shape-level heuristics, and votes for the answer choice that best completes the matrix.

Built for CS 7637 Knowledge-Based Artificial Intelligence at Georgia Tech.

---

## Contents

- [Overview](#overview)
- [Problem set](#problem-set)
- [How the agent works](#how-the-agent-works)
- [Project description](#project-description)
- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Results](#results)
- [References](#references)
- [Author](#author)
- [License](#license)

---

## Overview

| | |
|---|---|
| **Task** | Given a 2x2 or 3x3 matrix of images with one empty cell and a set of candidate images, select the candidate that correctly completes the matrix |
| **Approach** | Two heuristic families: shape-level features (Hu moments, aspect ratios, connected components) for advanced problems, and a pixel-level Visual Heuristic Methodology (Dark Pixel Ratio and Intersection Pixel Ratio) with weighted voting for general problems |
| **Dependencies** | Python standard library, NumPy, Pillow, OpenCV |
| **Learning** | None. The agent is fully deterministic and explainable; every decision can be traced to a measured comparison |
| **Problems** | 96, across Basic and Challenge sets B, C, D and E |

## Problem set

| Set | Matrix size | Count | Difficulty |
|---|---|---|---|
| Basic B | 2x2 | 12 | Introductory |
| Basic C | 3x3 | 12 | Intermediate |
| Basic D | 3x3 | 12 | Intermediate |
| Basic E | 3x3 | 12 | Advanced |
| Challenge B | 2x2 | 12 | Introductory, harder variants |
| Challenge C | 3x3 | 12 | Intermediate, harder variants |
| Challenge D | 3x3 | 12 | Intermediate, harder variants |
| Challenge E | 3x3 | 12 | Advanced, harder variants |

## How the agent works

```
Problem images (PNG)
        |
        v
  Perception (Pillow + OpenCV)
  Threshold, extract non-white pixel groups, compute shape features
        |
        v
  Route by problem class
        |
        +----> Advanced problems
        |      Hu moments, aspect ratios, connected components
        |      Capture higher-order geometric and abstract transformations
        |
        +----> General problems
               Visual Heuristic Methodology (Joyner et al., 2015)
               Dark Pixel Ratio (DPR) and Intersection Pixel Ratio (IPR)
               Compare each training pair against each candidate pair
        |
        v
  Weighted voting across all meaningful comparisons
        |
        v
  Answer: candidate with the most votes
```

The two measurements behind the general-problem heuristic:

| Measurement | Definition |
|---|---|
| **Dark Pixel Ratio (DPR)** | Dark pixels as a percentage of all pixels across a pair of adjacent matrix images |
| **Intersection Pixel Ratio (IPR)** | Dark pixels that overlap at the same coordinates, as a percentage of all dark pixels across a pair of adjacent matrix images |

## Project description

In this project, the Raven's Progressive Matrix (RPM) problems, which are psychometric problems, are attempted to be solved by an intelligent agent, developed in Python, which makes use of several heuristics. The RPM problems consist of 2x2 or 3x3 matrices with an empty slot (box) and a number of image answer choices provided. The agent's job is to programmatically identify the image answer choice that best fits the matrix and thereby solves the matrix problem. The agent code and heuristics should be generalizable enough such that when "new" or "unseen" problems are presented to the agent, the agent is able to solve them with a reasonable level of accuracy.</br> 
The PRM problems are 96 in total and have varying levels of complexity. The RPM problems are arranged in categories - Basic Problems B (2x2 matrix problems), Basic Problems C, D and E (3x3 matrix problems), Challenge Problems B (2x2 matrix problems), Challenge Problems C, D and E (3x3 matrix problems). The problems increase in complexity from the Basic Problems to the Challenge Problems. The only imported libraries used in the project, other than the packages in the Standard Python library are NumPy, Pillow and OpenCV. The Pillow library is mainly used for opening and manipulating images, while the OpenCV library is a computer vision library and is used to perform computer vision tasks, such as image recognition and 2D or 3D analysis to image motion tracking, image recognition, and the like. Computer vision tasks like these are required to solve the RPM problems. In solving the problems, the Python agent breaks down the problems into two broad categories - advanced problems and "general problems". For the advanced problems, heuristics such as Hu moments, aspect ratios, and connected components are used. The methodology used in the heuristics covered by this class of problems involve attempting to capture higher-order and complex relationships and geometric and abstract transformations in the matrix images in the problems such that most feasible option images, amongst the ones provided, can be identified to complete the matrices and solve the problems. For the "general problems", the agent uses a Visual Heuristic Methodology, described in the paper by [Joyner et al (2015)](https://computationalcreativity.net/iccc2015/proceedings/2_1Joyner.pdf), which is a pixel-level heuristic, consisting of pixel ratio measurements using the Dark Pixel Ratio (DPR) and Intersection Pixel Ratio (IPR). </br>
In this Visual Heuristic methodology, the matrix images are represented by collections of adjacent non-white pixels by the agent. The possibility of each of the image answer choices being the correct image that completes the matrix is computed by the agent. To do this, some measurements that pick up the relationship between each training pair (any two adjacent images in the matrix) are taken by the agent. These measurements are then compared, by the agent, with each one of the test/answer pairs and the grouping of any of the matrix images adjacent to the empty image position in the matrix and each image answer choice. Each of the meaningful comparisons casts a vote for the given image choice (which is represented by each given comparison) as the likely image that correct solves the matrix problem and completes the matrix. A weight measure is also assigned to each comparison, and this weight is directly associated with the perceived correlation of the option image with the images in the matrix. The image answer choice that has the most votes is selected as the answer that solves the matrix problem and completes the matrix.</br>
The agent uses these two measurements in this methodology: </br>
1. Dark pixel ratio (DPR) – this represents the percentage ratio of the number of pixels that are dark-colored and the total number of pixels that exist in a group of two adjacent matrix images. </br>
2. Intersection pixel ratio (IPR) – this represents the percentage ratio of the number of intersecting (on the same coordinates) pixels that are dark-colored and the total number of pixels that are dark-colored in a group of two adjacent matrix images.

## Getting started

### Requirements

Python 3.8 or later. The pinned dependencies are in `RPM-Project-Code54/requirements.txt`:

```
numpy==2.0.1
pillow==10.4.0
opencv-contrib-python-headless==4.10.0.84
```

```bash
cd RPM-Project-Code54
pip install -r requirements.txt
```

### Run the agent

```bash
python RavensProject.py
```

The runner loads every problem set listed in `ProblemSet.py`, calls `Agent.Solve()` on each problem, grades the answers with `RavensGrader.py`, and writes three files: `AgentAnswers.csv` (the agent's choice per problem), `ProblemResults.csv` (correct or incorrect per problem, with the expected answer), and `SetResults.csv` (the per-set summary).

## Repository structure

```
.
├── README.md
└── RPM-Project-Code54/
    ├── Agent.py               # The agent: perception, heuristics and weighted voting
    ├── RavensProject.py       # Entry point: runs every problem set and scores the agent
    ├── RavensGrader.py        # Grades the agent's answers against the answer key
    ├── ProblemSet.py          # Loads a problem set from the Problems/ folder
    ├── RavensProblem.py       # A single RPM problem: its matrix and answer choices
    ├── RavensFigure.py        # One figure (cell) within a problem
    ├── RavensObject.py        # One object within a figure
    ├── Problems/              # The 96 problems in eight sets
    │   ├── Basic Problems B/  ├── Basic Problems C/  ├── Basic Problems D/  ├── Basic Problems E/
    │   ├── Challenge Problems B/  ├── Challenge Problems C/  ├── Challenge Problems D/  └── Challenge Problems E/
    ├── requirements.txt       # Pinned dependencies
    ├── AgentAnswers.csv       # Output: the agent's answer per problem
    ├── ProblemResults.csv     # Output: correct or incorrect per problem
    └── SetResults.csv         # Output: per-set summary
```

## Results

Running `RavensProject.py` produces a per-problem and per-set breakdown in `ProblemResults.csv` and `SetResults.csv`. The per-set view is the useful one: it shows which relationship types the agent has genuinely captured and which it finds hardest. In general the agent is strongest on the 2x2 sets and on problems governed by progression and shape-transformation rules, and weakest on the compositional relationships in the E sets and the Challenge variants, where pixel-ratio heuristics have the least purchase.

## References

- Joyner, D., Bedwell, D., Graham, C., Lemmon, W., Martinez, O. and Goel, A. (2015). Using Human Computation to Acquire Novel Methods for Addressing Visual Analogy Problems on Intelligence Tests. *Proceedings of the Sixth International Conference on Computational Creativity (ICCC 2015)*. [PDF](https://computationalcreativity.net/iccc2015/proceedings/2_1Joyner.pdf)
- Raven, J. C. (1938). *Progressive Matrices: A Perceptual Test of Intelligence.* H. K. Lewis.
- Goel, A. K. and Davies, J. (2011). Artificial Intelligence. In R. J. Sternberg and S. B. Kaufman (Eds.), *The Cambridge Handbook of Intelligence.*

## Author

**Edidiong-Abasi Anwanane**  
MSc Computer Science, Georgia Institute of Technology  
[Portfolio](https://edidionga.github.io) · [LinkedIn](https://www.linkedin.com/in/edidiong-abasi-anwanane/) · [GitHub](https://github.com/EdidiongA)

## License

MIT. See [LICENSE](LICENSE).

<!-- If the course's academic-integrity policy restricts sharing solution code, consider adding a note here stating the repository is shared for portfolio purposes after course completion. -->
