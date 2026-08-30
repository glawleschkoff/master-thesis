This repository contains my Master's thesis.

<table>
  <tr>
    <td align="left"><strong>Title:</strong></td>
    <td align="left">Contextual Information Seeking for Active Inference</td>
  </tr>
  <tr>
    <td align="left"><strong>Author:</strong></td>
    <td align="left">Moritz Glawleschkoff, University Freiburg</td>
  </tr>
  <tr>
    <td align="left"><strong>Executive Supervisor:</strong></td>
    <td align="left">Dr. Sarah Schwöbel, TU Dresden</td>
  </tr>
  <tr>
    <td align="left"><strong>First Referee:</strong></td>
    <td align="left">Prof. Dr. Stefan Rotter, University Freiburg</td>
  </tr>
  <tr>
    <td align="left"><strong>Second Referee:</strong></td>
    <td align="left">Prof. Dr. Stefan Kiebel, TU Dresden</td>
  </tr>
  <tr>
    <td align="left"><strong>Date:</strong></td>
    <td align="left">9. September 2026</td>
  </tr>
</table>

**Abstract**  
Humans navigate complex environments by abstracting situational details into context states, enabling efficient perception and action. The Free Energy Principle (FEP) provides a framework for understanding this behavior, positing that agents minimize expected free energy, which naturally gives rise to both goal-directed action and information-seeking (curiosity). While epistemic behavior regarding immediate environmental states has been well-studied, contextual information-seeking—where agents actively resolve uncertainty about abstract situational rules—remains underexplored. To investigate these cognitive mechanisms, this thesis develops a computational behavioral model, the Context-GFE agent, capable of actively seeking information about an abstract context state. Grounded in Active Inference and formalized using Forney-style Factor Graphs (FFG), message passing algorithms are derived to enable the unified inference of past, present, and future belief distributions. The newly proposed Context-GFE agent is evaluated against three baseline models (BFE, Context-BFE, and GFE) in a simulated T-maze environment. Results demonstrate that the Context-GFE agent successfully resolves contextual uncertainty by planning ahead to receive informative observations, balancing epistemic and pragmatic drives. Overall, this model provides a mathematically grounded and testable hypothesis for studying contextual abstraction and cognitive control, facilitating future empirical validation through Bayesian model comparison.

**Reproducibility**  
* The figures 3.1, 3.2, and 3.3 in the thesis can be reproduced in the notebook `02_final_simulations/experiments.ipynb`.
* The PDF of the thesis is located in `03_thesis/build/main.pdf`.