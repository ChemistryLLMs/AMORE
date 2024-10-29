<h3 align="center"> Lost in Translation: Chemical LMs and the Misunderstanding of Molecule Structures </h3>
<p align="center">
  📃 Paper from EMNLP 2024 findings <br>
</p>

![poster](images/Lost_in_translation_poster.png)

## Experimental datasets
Experimental datasets are provided in the folder "data". CHEBI-20 folder contains test part of CHEBI-20 dataset and augmented ones, qm9 folder contains filteret isomeric molecules from qm9 dataset and its augmentations.Every dataset consist of "original" file and four augmented files, created by using augmentation code.

## Probing

### Augmentations
'./code/augmentation.py' creates 4 types of augmentations:
rdkit canonicalization
explicit addition of hydrogens
kekulization
replacement of cycle identifiers by random numbers
Full description is provided in the paper.

### Molecule description task evaluation
"dataset_augmentation_example" is example of augmentation data from original dataset and getting predictions.
"metrics" contains example of calculating ROUGE and METEOR between original gold descriptions and predictions on our augmented datasets.

### Last hiddden state evaluation
"eval_faiss.py" consist of functions for compute embeddings and metrics of all provided models.
"Last_hidden_state_MRR" contains example of calculating MRR metrics on our datasets and models
"last_hidden_state_reversed" contains example of calculating acc@1 and acc@5 metrics on our datasets and models.

### All hidden states evaluation
"hidden.py" consist of functions for compute embeddings of all layers and metrics of all provided models.
"hidden_states_scifive" contains example of calculating acc@1 and acc@5 metrics on our datasets.

##  References 
If you use our repository, please cite the following related paper:

```
@inproceedings{translation,
  title={Lost in Translation: Chemical LMs and the Misunderstanding of Molecule Structures},
  author={Ganeeva, Veronika and Sakhovskiy, Andrey and Khrabrov, Kuzma and Kadurin, Artur and Savchenko, Andrey and Tutubalina, Elena},
  booktitle={EMNLP 2024 Findings}
}
```
