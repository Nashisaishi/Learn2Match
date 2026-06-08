# Overcooked

Two-player fully cooperative game. From the popular videogame *Overcooked!*.

We provide a wrapper with core implementation from https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked.

A simple script to install the required Python packages (tested on CUDA 12 only):

```bash
# Download the code
git clone https://github.com/Social-RL/jax_pbt.git

# (Optional) Create a conda environment
# conda create -n <env_name> python=3.11
# conda activate <env_name>

# Install dependencies (tested on CUDA 12 only)
pip install -r jax_pbt/jax_pbt/env/overcooked/requirements.txt

# Install jax_pbt in editable mode
cd jax_pbt
pip install -e .
```



# TODO: readme and test continuous space