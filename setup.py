import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brokenegg_transformer", # Replace with your own username
    version="0.0.1",
    author="Katsuya Iida",
    author_email="katsuya.iida@gmail.com",
    description="A minimalist Transformer on Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brokenegg/transformer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['tensorflow>=2.1', 'sentencepiece']
)