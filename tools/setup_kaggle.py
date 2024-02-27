"""Simple script to setup python environment on kaggle.
"""
import subprocess

ENV_FILE = "/kaggle/temp/image_basics/environment/ibkaggle_environment_v0.yml"
ENV_NAME = "/root/mambaforge/envs/ibkaggle_v0"

install_statement = f"""
# install mambaforge
cd /kaggle/temp/
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh -b
cd /kaggle/working/

# create environment
export CONDA_ALWAYS_YES="true"
/root/mambaforge/bin/mamba env create -f {ENV_FILE}
source activate {ENV_NAME}
"""

subprocess.run(install_statement, shell=True, check=True)
