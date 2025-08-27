# Manual annotation of Robson criteria and obstetric entities: Inter-annotator agreement and initial NER models implementation

This repository contains scripts for training Named Entity Recognition (NER) models to identify specific obstetric information in Spanish-language Electronic Health Records. The models are based on several algorithms: Conditional Random Forest (CFR), RoBERTa, BERT, XLM-RoBERTa, and Llama-3.1-8B-Instruct. The identified entities include:

- Robson Criteria
- Obstetric complications and hemorrhage
- Delivery outcomes (vaginal birth or cesarean section)
- Medications (antibiotics and uterotonics, including their dosages and posologies)
- Personal information

The models are specifically designed to help analyze and extract crucial clinical data from unstructured text, which can be useful for research and clinical auditing.

For more details, check out the original paper here: [https://doi.org/10.1016/j.compbiomed.2025.110964](https://doi.org/10.1016/j.compbiomed.2025.110964)

## Annotation Guidelines
We developed annotation guidelines (Spanish) for this project. 

For more details, you can visit the full document here: [https://pln-disca-iimas.github.io/robson-criteria-guidelines/](https://pln-disca-iimas.github.io/robson-criteria-guidelines/)

## Web application prototype
Prototype of a web application to identify and extract clinical entities related to the Robson Criteria Classification (obstetric variables), among others. The application uses one of the trained Named Entity Recognition (NER) models, in addition to classifying the clinical note into one of the 10 Robson groups. You can visit the repository here: [https://github.com/orlandxrf/web-app-robson-criteria-classification](https://github.com/orlandxrf/web-app-robson-criteria-classification)


## Cite

```bibtex
@article{robson-criteria-ner-2025,
    title = {Manual annotation of Robson criteria and obstetric entities: Inter-annotator agreement and initial NER models implementation},
    journal = {Computers in Biology and Medicine},
    volume = {197},
    pages = {110964},
    year = {2025},
    issn = {0010-4825},
    doi = {https://doi.org/10.1016/j.compbiomed.2025.110964},
    url = {https://www.sciencedirect.com/science/article/pii/S0010482525013162},
    author = {Orlando Ramos-Flores and Helena Gómez-Adorno and Mónica Vazquez and Rodrigo {De Ita} and Juan-Manuel Mimiaga-Morales and Frida-Devi-Abigail Campos-Campechano and Manuel {García de Quevedo-Martínez} and María-Jose Aguilar-Sánchez and Marco-Antonio Ramírez-Mejía and Diego-Iván Jaramillo-Sánchez},
    keywords = {Robson criteria classification, Inter-annotation agreement, NER, BERT, RoBERTa, XLM-RoBERTa, Llama3.1-8B-instruct}
}
```

