This repository contains my Master's thesis.

<dl style="display: grid; grid-template-columns: max-content auto; row-gap: 4px; column-gap: 16px; margin: 16px 0;">
  <dt style="font-weight: bold;">Title:</dt>
  <dd style="margin: 0;">Contextual Information Seeking for Active Inference</dd>
  <dt style="font-weight: bold;">Author:</dt>
  <dd style="margin: 0;">Moritz Glawleschkoff, University Freiburg</dd>
  <dt style="font-weight: bold;">Executive Supervisor:</dt>
  <dd style="margin: 0;">Dr. Sarah Schwöbel, TU Dresden</dd>
  <dt style="font-weight: bold;">First Referee:</dt>
  <dd style="margin: 0;">Prof. Dr. Stefan Rotter, University Freiburg</dd>
  <dt style="font-weight: bold;">Second Referee:</dt>
  <dd style="margin: 0;">Prof. Dr. Stefan Kiebel, TU Dresden</dd>
  <dt style="font-weight: bold;">Date:</dt>
  <dd style="margin: 0;">9. September 2026</dd>
</dl>

**Abstract**  
Humans navigate complex environments by abstracting situational details into context states, enabling efficient perception and action. The Free Energy Principle (FEP) provides a framework for understanding this behavior, positing that agents minimize expected free energy, which naturally gives rise to both goal-directed action and information-seeking (curiosity). While epistemic behavior regarding immediate environmental states has been well-studied, contextual information-seeking—where agents actively resolve uncertainty about abstract situational rules—remains underexplored. To investigate these cognitive mechanisms, this thesis develops a computational behavioral model, the Context-GFE agent, capable of actively seeking information about an abstract context state. Grounded in Active Inference and formalized using Forney-style Factor Graphs (FFG), message passing algorithms are derived to enable the unified inference of past, present, and future belief distributions. The newly proposed Context-GFE agent is evaluated against three baseline models (BFE, Context-BFE, and GFE) in a simulated T-maze environment. Results demonstrate that the Context-GFE agent successfully resolves contextual uncertainty by planning ahead to receive informative observations, balancing epistemic and pragmatic drives. Overall, this model provides a mathematically grounded and testable hypothesis for studying contextual abstraction and cognitive control, facilitating future empirical validation through Bayesian model comparison.

**Reproducibility**  
* The figures 3.1, 3.2, and 3.3 in the thesis can be reproduced in the notebook `02_final_simulations/experiments.ipynb`.
* The PDF of the thesis is located in `03_thesis/build/main.pdf`.