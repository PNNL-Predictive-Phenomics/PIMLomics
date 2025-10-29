# PIMLomics

Description:

**PIMLomics** implements a physics-informed machine learning (PIML), data-driven framework (CellBox) to infer regulatory proteins in cyanobacterial metabolic networks under system perturbations. PIMLomics integrates multi-omics datasets and applies dimensional reduction (Independent Component Analysis with pymodulon) algorithms to build a dynamic de novo network from gene expressions and redox proteome dynamics. Designed for perturbation biology approaches, the workflow is adaptable for other organisms with available multi-omics data, providing a versatile tool for profiling metabolic processes and regulatory pathways in complex biological systems.

PIMLomics utilizes the CellBox PIML algorithm to model protein and transcription factor dynamics through an interaction-decay ordinary differential equation. Light-induced perturbations cause changes in protein and transcription factor counts, which decay back to steady-state values. This decay is modeled by combining single-protein decay rates with interactions within a network of proteins and transcription factors. The workflow optimizes equation parameters, such as the interaction matrix, to best fit experimental data. It analyzes both a full gene/protein model and a reduced model, using Independent Component Analysis (ICA) to cluster genes into modulons in an iterative loop until the model demonstrated the evidence of learning. PIMLomics then identifies significant protein and transcription factor interactions by comparing inferred parameters with measured abundance and redox proteomics data.

Key Features:

- Physics-informed Machine Learning analysis of multiomics data.
- Plotting and analysis of protein interaction networks.

Use Cases:

PIMLomics uses the Physics-informed Machine Learning package CellBox to guide investigations into complex protein interactions influenced by cell responses to probing perturbations.

Requirements:
- CellBox
- pymodulon

How To:
1) Construct inputs 
    1) Gene model
        1) Perturbations - Light and circadian time inputs from [Puszynska & O'Shea 2017](https://doi.org/10.7554/eLife.23210).
        2) TF log2FC in gene expression from reference condition (clearday 0.5h) (Log2FC calculated with DESeq2).
        3) Subset of circadian genes from [Markson et al 2013](https://doi.org/10.1016/j.cell.2013.11.005). Gene log2FC in gene expression from reference condition (clearday 0.5h) (Log2FC calculated with DESeq2).
    2) Module model
        1) Perturbations - Light and circadian time inputs from [Puszynska & O'Shea 2017](https://doi.org/10.7554/eLife.23210).
        2) TF log2FC gene expression values (Calculated using DESeq2)
        3) ICA signal averaged for each condition over three replicates.

2) Load input gene and iModulon models (ICA). Examples are provided in the Execution_files used for the analysis workflow. 
	1) Modify config files (config.json) to point to the respective file paths, and where values can be edited to tailor the CellBox simulation. Specific parameters include "n_protein_nodes": number of transcription factor, "n_activity_nodes": number of transcription factor + gene/modulon, "n_x": number of nodes in transcription factor + gene/modulon + perturbations,
	2) For n nodes considered over p perturbations:
		1) expr_matr.csv (p x n_x matrix) : matrix of expression values, one value for each node (transcription factor, gene/modulon, perturbation node) per each set of perturbation conditions. 
		2) node_index.csv (n_x list) : names for each node (without "," or delimiters), column list where number of lines correspond to the number of nodes in order of type (transcription factor, gene/modulon, perturbation node)
		3) pert_matr.csv (p x n_x matrix): matrix of perturbation values with the same shape as "expr_matr.csv". Zero values for none perturbed values with non-zero values for the perturbation nodes. 
		4) sample_order.csv (p list) : names for each perturbation (without "," or delimiters), column list where number of lines correspond to the number of perturbation cases considered.
	
3) Execute CellBox (example execution Slurm script provided "slurmscript.py" and within the "Execution_files" directory)
4) Execute analysis workflow:
	1) For single time point proteomics data
	2) Edit file paths in the jupyter notebooks to point to the CellBox output specified in "experiment_id" in config.json  (Analysis/FullPipelineForCellboxOutput_analysis.ipynb)
	3) Run notebook and edit plotting functions for names and proteomics datas
	4) Plotting functions can be edited to tailor visuals and output directories


## Citation

Johnson, C. G., Johnson, Z., Mackey, L. S., Li, X., Sadler, N. C., Zhang, T., ... & Cheung, M. S. (2025). Multi-Omics Reveals Temporal Scales of Carbon Metabolism in Synechococcus Elongatus PCC 7942 Under Light Disturbance. PRX Life, 3(3), 033017.

```
@article{l2dp-kw2t,
  title = {Multi-Omics Reveals Temporal Scales of Carbon Metabolism in Synechococcus Elongatus PCC 7942 Under Light Disturbance},
  author = {Johnson, Connah G. M. and Johnson, Zachary and Mackey, Liam S. and Li, Xiaolu and Sadler, Natalie C. and Zhang, Tong and Qian, Wei-Jun and Bohutskyi, Pavlo and Feng, Song and Cheung, Margaret S.},
  journal = {PRX Life},
  volume = {3},
  issue = {3},
  pages = {033017},
  numpages = {15},
  year = {2025},
  month = {Sep},
  publisher = {American Physical Society},
  doi = {10.1103/l2dp-kw2t},
  url = {https://link.aps.org/doi/10.1103/l2dp-kw2t}
}
```

## License

PIMLomics made freely available under the terms of Simplified BSD license.
Copyright 2025 Battelle Memorial Institute
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

<div align="center">

PACIFIC NORTHWEST NATIONAL LABORATORY  
operated by  
BATTELLE  
for the  
UNITED STATES DEPARTMENT OF ENERGY  
under Contract DE-AC05-76RL01830  

</div>


## Acknowledgments

This work was supported by the Predictive Phenomics Initiative, under the Laboratory Directed Research and Development Program at at the Pacific Northwest National Laboratory. PNNL is a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy under contract DE-AC05-76RL0 1830.
