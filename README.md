## **ENDOINSIGHTS-MACHINE LEARNING POWERED INSIGHTS FOR  BETTER ENDOMETRIOSIS  CARE**

EndoInsights is a research project that leverages deep learning for the improved ultrasound-based diagnosis of Endometriosis.
We use a hybrid CNN + Vision Transformer (ViT) model trained on the GLENDA dataset to classify and localize endometriosis from laparoscopic images.

🚀 Model Training

The model was trained on the GLENDA Dataset (Leibetseder et al., 2020
).

Training was conducted on Google Colab Pro using an NVIDIA T4 GPU.

Models are stored per epoch checkpoints for reproducibility and evaluation.

📂 Dataset

This project makes use of the GLENDA (Gynecologic Laparoscopy ENdometriosis DAtaset), comprising over 25,000 laparoscopic frames with region-based annotations.

Categories: Peritoneum, Ovary, Uterus, Deep Infiltrating Endometriosis (DIE), No Pathology

Annotations: Binary masks (COCO-style in v1.5)

Size: ~25k frames across 138 sequences

License: CC BY-NC 4.0
 (Non-commercial use only)

## 📑 Citation

If you use the GLENDA dataset, please cite:

```bibtex
@inproceedings{DBLP:conf/mmm/LeibetsederKSKK20,
    author    = {Andreas Leibetseder and
                 Sabrina Kletz and
                 Klaus Schoeffmann and
                 Simon Keckstein and
                 J{\"{o}}rg Keckstein},
    title     = {{GLENDA:} Gynecologic Laparoscopy Endometriosis Dataset},
    booktitle = {MultiMedia Modeling - 26th International Conference, {MMM} 2020, Daejeon,
                 South Korea, January 5-8, 2020, Proceedings, Part {II}},
    series    = {Lecture Notes in Computer Science},
    volume    = {11962},
    pages     = {439--450},
    publisher = {Springer},
    year      = {2020},
    doi       = {10.1007/978-3-030-37734-2_36}
}
```
⚙️ Installation

Clone this repository:

git clone https://github.com/ShallenCrissle/EndoInsights.git
cd EndoInsights


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


▶️ Usage
Training the Model
python training.py --epochs 30 --batch-size  --data ./dataset

Validating the model
python Validation.py

Running the Streamlit App
streamlit run app.py

📊 Results

Hybrid CNN + ViT architecture

Evaluation on GLENDA annotated frames

Models stored per epoch for reproducibility
