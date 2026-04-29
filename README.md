<p align="center">
  <img src="assets/banner.svg" alt="Advanced Fibrosis Calculator" width="100%"/>
</p>

<p align="center">
  <a href="https://laithomari.github.io/advanced_fibrosis_calculator/">
    <img src="https://img.shields.io/badge/Online_Calculator-Try_It_Now-2b6cb0?style=for-the-badge&logo=calculator&logoColor=white" alt="Online Calculator"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"/>
</p>

---

## Overview

A validated logistic regression model for predicting advanced fibrosis (F&ge;3) in MASLD patients with **indeterminate FIB-4 scores (1.3–2.67)**. The model uses **8 routinely available clinical variables** and was trained on 1,581 biopsy-confirmed patients from the NAFLD DB2 cohort.

**[Try the online calculator &rarr;](https://laithomari.github.io/advanced_fibrosis_calculator/)**

## Model Inputs

| Variable | Unit |
|:---------|:-----|
| Age | years |
| BMI | kg/m&sup2; |
| AST | U/L |
| ALT | U/L |
| GGT | U/L |
| Platelet count | &times;10&sup9;/L |
| Diabetes status | Yes / No |
| AST/ALT ratio | derived |

## Risk Stratification

| Category | Threshold | Suggested Action |
|:---------|:----------|:-----------------|
| **Low Risk** | &lt;25.6% | Advanced fibrosis unlikely; monitor in primary care |
| **Intermediate** | 25.6%–59.2% | Consider second-line testing (e.g., VCTE) |
| **High Risk** | &ge;59.2% | Refer to hepatology |

## Validation

| Cohort | N | AUROC (95% CI) |
|:-------|:--|:---------------|
| Internal (DB2) | 213 | 0.826 (0.762–0.880) |
| Asian External (LiveFbr) | 203 | 0.737 (0.668–0.801) |
| NHANES | 1,503 | 0.743 (0.698–0.788) |

## Citation

> *A Simple Model to Predict Advanced Fibrosis in MASLD Patients with Indeterminate FIB-4.* [Manuscript in preparation]

## License

MIT License. See [LICENSE](LICENSE) for details.
