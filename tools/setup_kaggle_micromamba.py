"""Simple script to setup python environment on kaggle.

~6min run time, preference towards the mambaforge version.
usage:
mkdir -p /kaggle/temp/
git -C /kaggle/temp/ clone https://github.com/hermangudjonson/image_basics.git

python /kaggle/temp/image_basics/tools/setup_kaggle_micromamba.py
source activate /root/micromamba/envs/ibkaggle_v0
"""
import subprocess

install_statement = """
# install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# create environment
export MAMBA_ALWAYS_YES="true"
/root/.local/bin/micromamba env create -f /kaggle/temp/image_basics/environment/ibkaggle_environment_v0.yml
"""

subprocess.run(install_statement, shell=True, check=True, executable="/bin/bash")
