# Extreme Events Engineering VIP

## Goals

Advance the fundamental science in the assessment of natural and man-made hazards (e.g. earthquakes, landslides, liquefaction, sea-level rise, hazards in tailing dams, heap leach pads, 
coal ash facilities) through novel developments in performance-based and risk engineering.

We combine performance-based engineering, reliability, machine learning, and artificial intelligence tools with advanced numerical simulations, and novel experimental procedures to 
advance the fundamental understanding of the interaction between geo-hazards and geotechnical systems under extreme loading events and climate change stressors. The ultimate goal is 
to make infrastructure systems and cities more resilient, saving lives, and reducing economic losses. Additionally, we address the issues that have led to recent catastrophic worldwide 
failures in the mining industry.

## Issues Involved or Addressed

Natural and man-made hazards, Geotechnical earthquake engineering,Advance numerical modeling and machine learning, performance-based design, risk engineering, mining geotechnics.

## Methods and Technologies

* Machine Learning
* Artificial Intelligence
* Numerical modeling (FEM, FDM, MPM, DEM)
* Advanced laboratory tests (static and cyclic)
* Material characterization techniques (e.g. image-based analyses)
* Programing (Matlab, Python, C++)


##Partner(s) and Sponsor(s)

* National Science Foundation
* GDOT
* IDOT
* Industry

## Project 1: Outlier Detection for Ground Motions

### Task 1: Identifying Features from Cyclic Data

For each test, we

* isolated the cycles within the graph of horizontal stress against vertical stress
* plotted graphs of horizontal stress against horizontal strain for each cycle
* Denoised the data

Then, we calculated the following indices for each cycle: secant shear modulus (Gsec), strain amplitude (SA), area in the curve (AC), and viscous energy damping ratio (VR). This can be seen in 
the ```CountCycles.py``` file.

### Task 2: Tangent Modulus Variation

In ```TangentialModulus.py```, we plotted the tangent modulus variation.

## Project 2: Using CPT Simulations to Predict Subsurface Conditions

* Take in data preprocessed from the TACC Supercomputers last semester and perform Regression on the data to predict the state
* Involves filtering the legal data from Column 2, masking over both X and Y, and then running train_test_split

R<sup>2</sup>	 Values
* Before filtering legal Data: 0.509
* After filtering legal Data: 
  * 10 estimators: 0.555
  * 15 estimators: 0.5676
  * 20 estimators: 0.5809
  * 50 estimators: 0.5892

This work is seen in the ```RandomForest.py``` file.

Thank you to Dr. Jorge Macedo and a team of PhD, graduate, and undergraduate students at Georgia Institute of Technology who worked with me on these projects.
