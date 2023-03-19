from setuptools import setup

setup(
    name="protein_diffusion",
    packages=[
        'data', 'model', 'inpainting'
    ],
    package_dir={
        'data': './data',
        'model': './model',
        'inpainting': './inpainting',
    },
)
