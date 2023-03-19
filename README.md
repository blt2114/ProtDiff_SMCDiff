# Diffusion Probabilistic Modeling of Protein Backbones in 3D for the Motif-Scaffolding Problem

Implementation for "Diffusion Probabilistic Modeling of Protein Backbones in 3D for the Motif-Scaffolding Problem" [paper link](https://arxiv.org/abs/2206.04119).
(Accepted at International Conference on Learning Representations 2023)

LICENSE: MIT

<img src="https://github.com/blt2114/protein_diffusion_share/blob/main/protdiff_traj.gif" width="300" height="342">


If you use our work then please cite
```
@article{trippe2022diffusion,
  title={Diffusion probabilistic modeling of protein backbones in 3D for the motif-scaffolding problem},
  author={Trippe, Brian L and Yim, Jason and Tischer, Doug and Broderick, Tamara and Baker, David and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2206.04119},
  year={2022}
}
```

## Installation

To install, you may use [miniconda](https://docs.conda.io/en/main/miniconda.html) (or anaconda).
Run the following to install a conda environment with the necessary dependencies.
```bash
conda env create -f protein_diffusion.yml
```

Next, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```

## Running
You can run examples of unconditional generation and motif-scaffolding with SMC-Diff with the included ipython notebook.
```
notebook/inference_example.ipynb
```

## Related works
This implementation has largely been superceded by [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion) and [RFdiffusion](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v2).

### Third party source code
Our model code adapts equivariant graph convolutionanal neural networks ([code link](https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py)).  If you adapt this code further, please credit them as well.
