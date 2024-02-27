"""Simple script to setup python environment on kaggle.

usage:
mkdir -p /kaggle/temp/
git -C /kaggle/temp/ clone https://github.com/hermangudjonson/image_basics.git

python setup_kaggle.py
source activate /root/mambaforge/envs/ibkaggle_v0
"""
import subprocess

ENV_FILE = "/kaggle/temp/image_basics/environment/ibkaggle_environment_v0.yml"

install_statement = f"""
# install mambaforge
cd /kaggle/temp/
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh -b
cd /kaggle/working/

# create environment
export CONDA_ALWAYS_YES="true"
/root/mambaforge/bin/mamba env create -f {ENV_FILE}
"""

subprocess.run(install_statement, shell=True, check=True)
