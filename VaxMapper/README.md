# README for RxNorm Term Getter

## Overview

The RxNorm Term Getter is a Python project designed to identify missing RxNorm terms related to vaccines that are not present in the current version of the Vaccine Ontology (VO). This tool utilizes the RxNorm API to fetch relevant data and compare it against the existing terms in the VO ontology.

## Project Structure

```
rxnorm-term-getter
├── src
│   ├── __init__.py
│   ├── rxnorm_term_getter.py
│   └── utils
│       ├── __init__.py
│       └── helpers.py
├── requirements.txt
├── README.md
└── main.py
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <https://github.com/warren-manuel/VaxMapper.git>
cd rxnorm-term-getter
pip install -r requirements.txt
```

## Usage

To run the script and identify missing RxNorm terms, execute the following command:

```bash
python main.py
```

You can also specify a custom path or URL to the VO ontology by modifying the `vo_path_or_url` parameter in the `main.py` file.

## Dependencies

The project requires the following Python packages:

- `pandas`
- `requests`
- `owlready2`

These packages are listed in the `requirements.txt` file.
