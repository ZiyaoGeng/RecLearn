import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="reclearn",
  version="1.0.16",
  author="Ziyao Geng",
  author_email="zggzy1996@163.com",
  description="A simple package about learning recommendation",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0",
  packages=setuptools.find_packages(),
  python_requires=">=3.6",
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
  license="MIT",
)