# Rashomon Restricted Ambiguity Averse Active Learning (RRAA-AL)

🚧 This repository is currently under construction. 🚧

Updates are ongoing as part of iterative version control and management.

## Abstract
Collecting labeled data for training machine learning models is often costly and time-consuming. Active learning addresses these constraints by adaptively and strategically selecting the most informative observations for labeling. However, the current active learning literature does not account for model ambiguity and the possibility that several near-optimal models may fit the data well, a phenomenon termed the Rashomon Effect. When selecting the most informative candidate observation for labelling, the Rashomon Effect may suggest different candidate observations. Which observation, then, should an analyst query?

In this work, we propose a novel active learning algorithm, that addresses both this predictive multiplicity and the core issue of model ambiguity. Our approach enumerates the Rashomon set of near-optimal models and weights the selection metric by the posterior model probability. We then select the observation with the highest uncertainty from any model in the Rashomon set, embodying a conservative, ambiguity-averse strategy. This ensures a “best worst case” decision-making process restricted to the most plausible models, providing a robust method for selecting informative observations under model uncertainty. We term this procedure Rashomon Restricted Ambiguity Averse Active Learning (RRAA-AL).

## Setup

## Implementation

## Running

### Simulated data

### Application dataset
