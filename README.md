<p  align="center">
  <img src='logo.png' width='400'>
</p>

# CounterReview Logic: Automatic Reviewers Fail to Detect Faulty Reasoning in Research Papers -- A New Counterfactual Evaluation Framework
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](TBA)
[![License](https://img.shields.io/github/license/UKPLab/counter-review-logic)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/counter-review-logic/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/counter-review-logic/actions/workflows/main.yml)

This is the code associated with the paper "Automatic Reviewers Fail to Detect Faulty Reasoning in Research Papers: A New Counterfactual Evaluation Framework".

> **Abstract:** Large Language Models (LLMs) have great potential to accelerate and support scholarly peer review and are increasingly used as fully automatic review generators (ARGs). However, potential biases and systematic errors may pose significant risks to scientific integrity; understanding the specific capabilities and limitations of state-of-the-art ARGs is essential. We focus on a core reviewing skill that underpins high-quality peer review: detecting faulty research logic. This involves evaluating the internal consistency between a paper’s results, interpretations, and claims. We present a fully automated counterfactual evaluation framework that isolates and tests this skill under controlled conditions. Testing a range of ARG approaches, we find that, contrary to expectation, flaws in research logic have no significant effect on their output reviews. Based on our findings, we derive three actionable recommendations for future work and release our counterfactual dataset and evaluation framework publicly.
> 

Contact person: [Nils Dycke](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/staff_ukp/ukp_home_content_staff_1_details_109248.en.jsp) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

## Getting Started

> **DO NOT CLONE OR FORK**

If you want to set up this template:

1. Request a repository on UKP Lab's GitHub by following the standard procedure on the wiki. It will install the template directly. Alternatively, set it up in your personal GitHub account by clicking **[Use this template](https://github.com/rochacbruno/python-project-template/generate)**.
2. Wait until the first run of CI finishes. Github Actions will commit to your new repo with a "✅ Ready to clone and code" message.
3. Delete optional files: 
    - If you don't need automatic documentation generation, you can delete folder `docs`, file `.github\workflows\docs.yml` and `mkdocs.yml`
    - If you don't want automatic testing, you can delete folder `tests` and file `.github\workflows\tests.yml`
    - If you do not wish to have a project page, delete folder `static` and files `.nojekyll`, `index.html`
4. Prepare a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install .
pip install -r requirements-dev.txt # Only needed for development
```
5. Adapt anything else (for example this file) to your project. 

6. Read the file [ABOUT_THIS_TEMPLATE.md](ABOUT_THIS_TEMPLATE.md)  for more information about development.

## Usage

### Using the classes

To import classes/methods of `counter_review_logic` from inside the package itself you can use relative imports: 

```py
from .base import BaseClass # Notice how I omit the package name

BaseClass().something()
```

To import classes/methods from outside the package (e.g. when you want to use the package in some other project) you can instead refer to the package name:

```py
from counter_review_logic import BaseClass # Notice how I omit the file name
from counter_review_logic.subpackage import SubPackageClass # Here it's necessary because it's a subpackage

BaseClass().something()
SubPackageClass().something()
```

### Using scripts

This is how you can use `counter_review_logic` from command line:

```bash
$ python -m counter_review_logic
```

### Expected results

After running the experiments, you should expect the following results:

(Feel free to describe your expected results here...)

### Parameter description

* `x, --xxxx`: This parameter does something nice

* ...

* `z, --zzzz`: This parameter does something even nicer

## Development

Read the FAQs in [ABOUT_THIS_TEMPLATE.md](ABOUT_THIS_TEMPLATE.md) to learn more about how this template works and where you should put your classes & methods. Make sure you've correctly installed `requirements-dev.txt` dependencies

## Cite

Please use the following citation:

```
@InProceedings{smith:20xx:CONFERENCE_TITLE,
  author    = {Smith, John},
  title     = {My Paper Title},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
