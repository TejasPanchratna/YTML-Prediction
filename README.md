# YTML-Prediction

**YTML-Prediction** is a Machine Learning project designed to predict YouTube video performance using both textual and numerical data.  
The project demonstrates data collection through the YouTube Data API, exploratory data analysis, model training, and deployment integration.

---

## Overview

The dataset used in this project was collected using the official **YouTube Data API** provided by Google.  
A total of approximately **6,000+ YouTube videos** were gathered, containing various features such as video title, tags, description, and engagement metrics (views, likes, comments, etc.).

Multiple datasets are included in this repository:
- **Raw data** – directly fetched from the API  
- **Cleaned data** – processed and refined  
- **Merged data** – combined and ready for modeling

Both **classification** and **regression** models were trained to predict performance metrics and categorical outcomes.

---

## Model Integration and Deployment

This trained model is deployed as part of my portfolio website, available in the repository:  
**[MyPortfolioWebsite](https://github.com/TejasPanchratna/MYWEBSITE.git)**

That project integrates:
- A **NestJS** backend  
- A **React** frontend  
- This **machine learning model**, served through an API for real-time predictions

To maintain modularity, this repository also includes a **FastAPI** implementation for local testing and standalone use.

---

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted using Python libraries such as:
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**

The analysis focused on understanding the relationships between metadata features and engagement metrics.  
All EDA visuals, distribution plots, and correlation heatmaps are included in the `images/` directory and are also displayed on the portfolio website.

---

## Repository Structure

| Folder/File | Description |
|--------------|-------------|
| `Datasets/` | Contains raw, cleaned, and merged datasets |
| `CodeFiles/` | Jupyter notebooks for training and evaluation |
| `RawEDA/` | Plots, charts, and EDA visualizations |
| `prediction_api/app.py` | FastAPI application for serving model predictions |
| `prediction_api/requirements.txt` | List of dependencies and environment setup instructions |
| `prediction_api/models` | contains .joblib files of classifier, regressor and the scaler |
| `prediction_api/venv` | Virtual environemt for local execution |
| `prediction_api/__pycache__` | Auto-generated Python bytecode cache (can be ignored in Git) |

---

## Technologies Used

- Python 3.x  
- NumPy, Pandas, Matplotlib, Seaborn  
- Scikit-learn  
- FastAPI  
- Jupyter Notebook  
- Virtual environment for dependency isolation

---

## Future Work

Planned improvements include:
1. **Version 2 (V2):** Image-based model leveraging thumbnail data for visual feature extraction.  
2. **Version 3 (V3):** Multimodal integration combining textual, numerical, and visual data.  
3. Cloud-based deployment for scalability and public accessibility.  

---

## Author

Developed and maintained by **[Tejas Panchratna](https://github.com/TejasPanchratna)**.  
For questions, discussions, or collaboration inquiries, feel free to open an issue or contact through the linked GitHub profile.
