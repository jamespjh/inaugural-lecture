# Tasks required for the MVP

Tag human means task will need human involvement
Tag agent means task can have significant agent contribution
Tag with both human and agent where significant collaboration needed
Tag learning means task could be done by agent, but will be done by human for pedagogical reasons

## Set up quarto extensions for reusable patterns

Tags: agent, MVP

Templates for:

- video callout with alt-text and image credit
- image include with alt-text and image credit
- two-column layout

With variables for the parameters

Move column width styling on the columns currently on the slides into the css stylesheet

## Clean up all the text as far as placeholders for each graphic

Tags: human, MVP

For each slide:

- Break it up if it is too long, and renumber appropriately
- Select text to use for each slide
- Format appropriately using the quarto extensions

## Refactor the demo code so that solver is independent of law

Tags: agent, MVP

- New command line parameter law to select constant velocity vs gravity vs fits vs other future choices
- The fitted laws should error out if used with solvers other than pure python Euler steps
- Create a constant velocity law, and use it to generate a figure for slide 1.1.3

## Make trail slide with Euler and LSODA to illustrate non-closing curve

Tags: agent, MVP

- Add to the list of images to be generated for the slides
- Hook into current slide 1.1.6 which will need splitting up
- It might not be 1.1.6 by the time you get to this.

## Make it so the RNG seed can be passed as a parameter

Tags: agent, MVP

- Different engines handle RNG seeds in different ways, so that needs to be handled carefully

## Refactor the demo code so that fit training is independent of running a simulation with a fit

Tags: agent, MVP

- New entry cli point for training
  - pass scenario to be used with different seeds to generate training data
- Save the parameters to the fit in an appropriate file
  - for the simple parametric fit, this will just be a yaml file with parameters
  - for the GP, choose a format appropriate to sklearn
- When using a trained law, new command line parameter --model-data which points to the fit file
  - error out if not present and one of these laws is used
- No longer always retrain when the law is invoked

## Benchmark figure showing performance of the gravity law with each engine

Tags: agent, MVP

- Copy in the more sophisticated benchmarking and matrix abstraction code from https://github.com/jamespjh/stubs/blob/main/languages/python/multiply/src/multiply/array_abstraction.py and https://github.com/jamespjh/stubs/blob/main/languages/python/multiply/src/multiply/benchmark.py over the top of our existing files and refactor accordingly
- Modify the --benchmark switch to just run a single step of the law with the scenario specified
- Make a new CLI entry point to run a benchmark over the range of engines and model sizes
- Take inspiration from https://github.com/jamespjh/stubs/blob/main/languages/python/multiply/src/multiply/notebooks/cuda.ipynb 
- For now, will need to run on my GPU node and/or apple silicon laptop and commit the figure rather than
being part of CI/CD pipeline

## Make the number of bodies for the scatter scenario be specifiable as a command line parameter

Tags: agent, MVP

- Error out if given for non-variable-body-count scenarios

## Choose a nice figure from the scatter scenario

Tags: agent, human, MVP

- Make a local folder to contain the candidate models outputs 
- Make a yaml file to generate outputs for each of several candidates using the generate-figures entry point
- It should use the scatter scenario with different RNG seeds and different numbers of bodies
- It should output still figures, using the trail visualisation
- Human will review these and pick a nice one

## Implement the Boids law

Tags: agent, human, MVP

- Take the numpy boids code from https://github.com/jamespjh/bad-boids/blob/better_boids/boids.py 
- Make a law file
- Set up an appropriate scenario to visualise - maybe scatter will work (human needed)
- Put the plot into slide 1.1.7

## Record a video of completing some task using agentic assistance for the final slide

Tags: human, MVP

- Use one of the non-MVP tasks listed below

# Tasks beyond the MVP

## Add an option to colorise the bodies in the scatter scenario

Tags: agent

- Specify the list of colornames in a helper python file to the viz file
- Choose colornames appropriate to the sun and planets of earth's solar system N=0..8
- Use white for N>=9
- cli option --colored to trigger

## Add three-D visualisation

Tags: agent
- Try some scatter scenarios with three dimensions
- add cli parameter for dimension
- Add appropriate viz code

## Add a fitted law with a broader function than the power law example

Tags: agent, human
- Use a multi-level-perceptron
- Still depends only on distance between particle pairs
- Use torch
- Choose an appropriate file format to save the fitted model
- Modify the slides to use this example as well as the power law example (human needed)

## Add a fitted law which depends on relative velocity and displacement not just distance

Tags: agent, human, learning
- Use a multi-level-perceptron
- Still applied pairwise between particles
- No assumption of equal-and-opposite-reaction
- Aim is to be able to fit both gravity and the boids

## Performance benchmark the ODE solver as well as the law

Tags: agent, human, learning

- Need to get autodiff working properly for Diffrax solve