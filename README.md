# Evaluating the DoC-Forest tool for Classifying the State of Consciousness in a Completely Locked-In Syndrome Patient

Kirsten, Oliver and Bogdan, Martin and Adama, Sophie

This paper ([PDF, ](documents/paper_icispc_2023.pdf) [doi.org/10.1109/ICISPC59567.2023.00015]( doi.org/10.1109/ICISPC59567.2023.00015)) has been presented at the 2023 7th International Conference on Imaging, Signal Processing and Communications (ICISPC).

## Abstract

This paper shows how the DoC-Forest method can
be used for classifying the state of consciousness in a Completely
Locked-In Syndrome (CLIS) patient based on electrophysiological measurements of the patients’ brain activity. The DoC-Forest
is a Machine Learning (ML) based method to classify whether
patients with a Disorder of Consciousness (DoC) are either in the
minimally conscious state (MCS) or have the unresponsive wakefulness syndrome (UWS). As labeled training data is unavailable
for CLIS patients, Stereoelectroencephalography (sEEG) data
obtained from patients undergoing anesthesia was used to train
an ensemble of decision trees to classify the state of consciousness.
The model achieves a predictive area under the curve (AUC) of
0.73 on out-of-sample data during training. The most important
features for training the model on the anesthesia dataset were
determined, and an explainable ML algorithm was used to
quantify how features contributed to forming the predictions
for the CLIS patient. For the CLIS dataset comprising ECoG
measurements from one CLIS patient, there was a time period
where the CLIS patient clearly was conscious. Based on the
predictions of the model trained on the anesthesia dataset,
this period was clearly identifiable, indicating that despite the
differences in the underlying conditions (CLIS and anesthesia)
and the different measurement methods (sEEG and ECoG) the
model was able to extract robust patterns predictive of the state
of consciousness.

## Results
- Notebook [evaluate_doc_forest_epochs.ipynb](https://github.com/randomrain101/doc-forest_clis_anesthesia/blob/main/notebooks/doc_for_clis/evaluate_doc_forest_epochs.ipynb) showing performance and feature importance on anesthesia data

- Notebook [predict_clis_with_doc_forest.ipynb](https://github.com/randomrain101/doc-forest_clis_anesthesia/blob/main/notebooks/doc_for_clis/predict_clis_with_doc_forest.ipynb) showing predictions and feature contributions for CLIS data

## Cite

    @inproceedings{kirsten2023evaluating,
        title={Evaluating the DoC-Forest tool for Classifying the State of Consciousness in a Completely Locked-In Syndrome Patient},
        author={Kirsten, Oliver and Bogdan, Martin and Adama, Sophie},
        booktitle={2023 7th International Conference on Imaging, Signal Processing and Communications (ICISPC)},
        pages={37--41},
        year={2023},
        organization={IEEE}
    }

## License

All source code is made available under a GNU AFFERO GENERAL PUBLIC LICENSE Version 3 license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text and presentation slides are not open source. The authors reserve the rights to the
article content.