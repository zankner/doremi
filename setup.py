from setuptools import setup, find_packages

setup(
    name='doremi',
    version='0.0.1',
    description='DoReMi Data reweighting algorithm',
    url='https://github.com/sangmichaelxie/doremi',
    author='Sang Michael Xie',
    author_email='xie@cs.stanford.edu',
    packages=find_packages('.'),
    install_requires=[
        'tokenizers==0.13.2',
        'transformers==4.27.2',
        'mosaicml-streaming',
        'torch==2.0.1',
        'torchvision',
        'datasets==2.10.1',
        'zstandard',
        'accelerate==0.18.0',
        'bitsandbytes==0.37.2',
        'evaluate==0.4.0',
        'scikit-learn==1.2.2',
        'wandb==0.14.0',
        'xformers==0.0.17',
        'tqdm',
    ],
)
