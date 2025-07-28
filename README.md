This project employs a complete stack machine learning pipeline to forecast the structural column type given the engineering input parameters such as PCD, weight, elevation, number of legs, wind speed, and seismic zone. It utilizes sophisticated ensemble techniques—Random Forest and XGBoost in stacked architecture—to provide secure and precise predictions.

The solution comprises:

-Data preprocessing, label encoding, scaling

-Class imbalance handling with SMOTE

-Ensemble training with StackingClassifier

-Multi-class ROC-AUC plot

-A Tkinter GUI-based user interface

SQLite database for storing predictions

--Features
 Stacked ML Model with RandomForest and XGBoost

 Data Preprocessing with Label Encoding, Scaling, and SMOTE

Multi-Class ROC Curve Visualization

Interactive GUI using Tkinter

Data Persistence with SQLite

Robust input validations & GUI feedback

 Tech Stack
Python

Pandas, NumPy, Matplotlib, Seaborn

scikit-learn, XGBoost, imbalanced-learn (SMOTE)

SQLite3

Tkinter (GUI)

Project Structure
Project Root
│
├── FinalDSCode.py         # Main Python script
├── augmented_dataset.xlsx # Input dataset (put this in D://SY4THSEM/DS/)
├── column_data.db         # SQLite DB (auto-created)
 How to Run
Clone this repository.

Install required packages:


pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn openpyxl

Run the script:

python FinalDSCode.py
The GUI will appear.
Put values and click Predict & Save.

Screenshots:
DATA sets before cleaning

<img width="309" height="252" alt="image" src="https://github.com/user-attachments/assets/0b773b7c-eda9-4a8a-ab04-57ba3250fd03" />

DATA sets after cleaning:

<img width="257" height="231" alt="image" src="https://github.com/user-attachments/assets/2cf51ae8-c090-4075-8416-a5784c56fb6d" />

 Confusion Matrix:
 
 <img width="299" height="235" alt="image" src="https://github.com/user-attachments/assets/302e246b-11aa-4c00-8fbd-04b3a6b49dd9" />
ROC curve:

<img width="331" height="163" alt="image" src="https://github.com/user-attachments/assets/12342aba-e4d4-4973-9c42-4761f180c344" />



--Future Improvements
Integrate PDF/CSV export of predictions

Add support for re-training from GUI

Add logging and exception tracking

