# Omiye
Omiye is a word-based dialect identification model developed to investigate the computational identification of Supare, a Yoruba variety spoken in Àkókò South West, Ondo State, Nigeria.
The project has two complementary components:
Omiye — a language/dialect identification model trained to distinguish Supare from Hausa, Igbo, and Yoruba.
External LID benchmark — an evaluation of pretrained AfroLID and CommonLingua to determine whether existing multilingual language identification systems classify Supare as Yoruba when Supare is not an explicit output class.
The distinction is important: Omiye treats Supare as a separate class, whereas the external models are evaluated in their pretrained form and therefore determine which of their existing language categories best matches Supare.
To run the code, the code model training.py is responsible for training Omiye, test_afrolid.py is used to test Afrolid inference,benchmark_dialect.py to benchmark.

#Benchmarking Policy
The external benchmark follows these rules:
No external model is retrained on the Supare evaluation data.
No external model is fine-tuned on the test set.
All models receive the same held-out test texts.
Gold labels come from the original test dataset.
Omiye predictions are not used as gold labels for external models.
External predictions are generated independently.

#Reproducing the Experiment
Prepare the environment
Create and activate a Python environment:
python -m venv benchmark_env
.\benchmark_env\Scripts\Activate.ps1

Install the required dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn joblib

Additional dependencies required by the external models should be installed according to their respective model documentation.













