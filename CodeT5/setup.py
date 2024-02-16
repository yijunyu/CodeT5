from setuptools import setup, find_packages, find_namespace_packages
import platform
import os
os.environ['CURL_CA_BUNDLE'] = ''

install_requires = [
  "accelerate==0.20.3",
]

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")
DEPENDENCY_LINKS.append("git+https://github.com/huggingface/transformers.git")
DEPENDENCY_LINKS.append("git+https://github.com/huggingface/peft.git")
    
setup(
  name = 'codet5',
  version = "1.0",
  py_modules = ['codet5'],
  author = 'Nghi D. Q. Bui',
  long_description=open("README.md", "r", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  keywords="AI4Code, Code Intelligence, Generative AI, Deep Learning, Library, PyTorch, HuggingFace",
  license="Apache 2.0",
  url = 'https://github.com/Salesforce/CodeT5',
  packages=find_packages(where=".", exclude=["tests", "assets", "datasets"]),
  package_data={'codet5': ['configs/*']},
  install_requires=install_requires,
  include_package_data=True,
  zip_safe=False,
  python_requires=">=3.8.0",
  dependency_links=DEPENDENCY_LINKS,
)
