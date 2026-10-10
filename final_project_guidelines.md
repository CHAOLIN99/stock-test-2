# Final Project

## Objective
To creatively apply knowledge gained through the course of the semester to a substantial learning problem of your own choosing.

## Deliverables
There are two deliverables for the final:
1. An Excel spreadsheet (or CSV file) that shows the total amount of time you spent on your final, broken down by day.
2. A PDF writeup of your project (two pages is required, can be longer if you'd like).

## Grading Standards
* Your final project counts as 20% of your overall grade.
* Grading is divided into two parts: 
  * **80%** of your final project grade is based on the number of hours you spent.
  * **20%** is based on your writeup.
* For the number of hours, the total number of hours will be divided by 35, then multiplied by 100 (and capped at $100\%$). This will be your percentage (e.g., $35\text{ hours} = 100\%$, $17.5\text{ hours} = 50\%$, etc.).
* Your writeup will be evaluated primarily based on the quality of your writing, although points may be assigned based on the quality of your project.
* **Note:** No late submissions are possible for this project, because it is done in lieu of the final exam.

## Description
For your final project, you should execute a substantial project of your own choosing. You will turn in a single writeup (in PDF format only, please!). Your writeup can be structured in whatever way makes sense for your project, but see below for some possible outlines.

Your project will be graded more on effort than results. As stated in class, it is better to swing for the fences and miss than to take on a simple, safe project. It is therefore very important that your final time log clearly convey the scope of your efforts. Serious effort is expected, and your writeup should reflect that.

## Requirements for the Time Log
For the time log, you must document the time you spent (on a daily basis) along with a simple description of your activities during that time. If you do not document your time, it will not count. In other words, it is not acceptable to claim that you spent 35 hours on your project without a time log to back it up. No excuses will be accepted for this requirement.

Example time log format:
- 8/11 - 1 hour - read AlphaGo paper
- 8/12 - 2 hours - downloaded and cleaned data
- 8/21 - 4 hours - found AlphaGo code
- 8/24 - 1 hour - implemented game logic
- 9/17 - 2 hours - worked on self-play engine
- 9/18 - 1 hour - worked on self-play engine
- 10/1 - 2 hours - started training
- ... etc.

### Additional Time Requirements:
- You may not count any more than **5 hours** of research and reading.
- You may not count any more than **15 hours** of "prep work" (dataset preparation, collection, cleaning, or wrestling with getting a simulator/model working for a deep RL project, etc.).
- At least **20 hours** must involve designing, testing, and iterating deep learning-based models, analyzing results, experimenting, etc.
- You do not get extra credit for more than 35 hours.

## Requirements for the Writeup
Your writeup serves to inform the instructor about what you did, and simply needs to describe your project. You should describe:
- The problem you set out to solve
- The exploratory data analysis you did
- Your technical approach
- Your results
It should be at least **2 pages**.

## Possible Project Ideas
Many different kinds of final projects are possible. A few examples include:
- Learning how to render a scene based on examples of position and lighting
- Learning which way is "up" in a photo (useful for drone odometry)
- Training an HTTP server to predict which web pages a user will likely visit next
- Training an earthquake predictor
- Using GANs to turn rendered faces into something more realistic (avoiding the "uncanny valley")
- Transforming Minecraft into a more realistic looking game with DNN post-processing
- Using style transfer on a network trained for facial recognition (to identify and accentuate facial characteristics)
- Using RGB+Depth datasets to improve geometric plausibility of GANs

The project can involve any application area, but the core challenge must be tackled using some sort of deep learning.

The best projects involve a new, substantive idea and novel dataset. It may also be acceptable to use vanilla DNN techniques on a novel dataset, as long as you demonstrate significant effort in the "science" of the project – evaluating results, exploring topologies, thinking hard about how to train, and careful test/training evaluation. Implementing a state-of-the-art method from the literature is also acceptable after prior approval.

## Notes & Guiding Questions
You are welcome to use any publicly available code on the internet to help you. Consider answering the following questions in your report:

### 1. Discussion of the Dataset
- Where did it come from? Who published it?
- Who cares about this data?

### 2. Discussion of the Problem to be Solved
- Is this a classification problem? A regression problem?
- Is it supervised? Unsupervised?
- What sort of background knowledge do you have to bring to bear on this problem?
- What other approaches have been tried? How did they fare?

### 3. Exploration of the Dataset
- What does the data include? What patterns do you see before coding?
- Any visualizations about the data you deem relevant.

### 4. Technical Description of Your Approach
- Background on the approach
- Description of the model and inference/training algorithm used
- How you partitioned your data into a test/training split
- Number of parameters, optimizer used, and topology choice
- Source of any pre-trained weights

### 5. Analysis of Results
- What was your final RMSE on your private test/training split?
- Did you overfit? How do you know?
- Did you iterate your design? Why or why not?
- Did you solve (or make progress on) the problem you set out to solve?

## Possible Sources of Interesting Datasets
- Crowdflower
- KDD Cup
- UCI Repository
- Kaggle (current and past)
- Data.gov
- AWS
- World Bank
- BYU CS478 Datasets
- Data | Utah.gov
- Google Research
- BYU DSC Competition