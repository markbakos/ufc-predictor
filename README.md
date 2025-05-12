# UFC Predictor

<img src="https://wakatime.com/badge/user/7a2d5960-3199-4705-8543-83755e2b4d0c/project/27c5d40f-233b-4a88-9484-09663e1e5926.svg" alt="Time spent in project" title="Time spent in project" />

## Improved model and scraper

**This repository is outdated** <br>
Reworked scraper and prediction model available at new repository [Advanced UFC Analyzer](https://github.com/markbakos/advanced-ufc-analyzer)

## Table of Contents

+ [About](#about)
+ [Features](#features)
+ [Requirements](#requirements)
+ [Installation](#installation)
+ [Current Model](#model)
+ [Contributing](#contributing)
+ [Contact](#contact)

## About <a name = "about"></a>

This project uses machine learning to predict the winner between 2 UFC fighters using historical fight data and fighter statistics. Using a supervised learning approach, the model is built on a neural network architecture that processes attributes of fighters, including height, weight, age, striking, grappling statistics and etc...
By analyzing patterns and relationships in the historical matchups, it gives greater weight to features of fighters that have been more apparent in past winners.
<br><br>
We do some preprocessing on the data to make sure the quality of it is as good and not overfitted, then duplicate the fight data, swapping `fighter1` for `fighter2`, balancing out the dataset, avoiding bias from the model so it focuses on the comparative features instead of the order of the fighters. 
When the data is ready, we train the model, we save it in the `models/` directory so that we can reuse it without training the model every time a prediction is to be made.

<br>For the dataset, I used [jansen88's UFC data scraper](https://github.com/jansen88/ufc-data), follow the instructions there.


## Features <a name = "features"></a>

**Data Handling:**

- Extracted useful fighter statistics (height, weight, age, striking and grappling statistics)
- Calculated advantages (e.g. height/reach advantage, striking efficiency)
- Encoded fighter stances and weight classes numerically with **LabelEncoder**
  
**Machine Learning Model:**

- Built a neural network using **TensorFlow/Keras** in **Python** to predict fight outcomes
- With the preprocessed data, train the model and valide its performance to make sure we get reliable predictions 
- Saved the model in `models/` and load it from there

**Prediction:**

- Using the trained model, it predicts the favored fighter based on their statistics, providing a confidence score

**API:**

- I used **FastAPI** to make the backend to handle requests for predictions.

**UI:**

- Created the responsive frontend using **Vue**, **TypeScript** and **TailwindCSS**
- Users can input the names of two fighters and send the request to the API from the site

## Requirements <a name = "requirements"></a>

### Prerequisites
1. **Python 3.10 or higher**: Install from [python.org](https://www.python.org/downloads/).
2. **pip**: Python package manager (comes with Python installations).
3. **Node.js (v14 or later)**: Install from [nodejs.org](https://nodejs.org/en/download/package-manager)

### Python Dependencies

Install the required Python packages from `requirements.txt` found in the root folder.

```
pip install -r requirements.txt
```

## Installation <a name = "installation"></a>

1. **Clone the repository**
```
 git clone https://github.com/markbakos/ufc-predictor.git
 cd ufc-predictor
```

2. **Install dependencies for frontend**
```
cd client
npm install
```

3. **Start the development server**
- Frontend:
```
  cd client
  npm run dev
```

- Backend:
```
  # Make sure you are in root folder
  source venv/bin/activate
  uvicorn app.server:app
```

4. **Include dataset**

Place your dataset into the `app/data/` folder. Make sure it is called `complete_ufc_data.csv`

5. **Open the app in your browser**<br>

Navigate to [http://localhost:5173](http://localhost:5173) or the address provided in your terminal to use the app.

## Current Model <a name = "model"></a>

The model currently shows accuracy of around **~66.5%** for blind data.

<img src="https://github.com/markbakos/ufc-predictor/blob/master/img/plot-v1.1.png?raw=true" alt="Training and Validation Accuracy" />

<img src="https://github.com/markbakos/ufc-predictor/blob/master/img/confusion_matrix-v1.1.png?raw=true" alt="Confusion Matrix" />


## Planned Features <a name = "planned"></a>

The current model only compares the fighters' physical and fighting statistics, it still lacks many features, including: <br>
(These might not improve the accuracy of the model, only a theory)

### Implemented

- Fighter's win streaks, win rates and momentum (v1.1) ✔️ <br>
Improved overall accuracy by about ~0.5% - ~0.8%. 

### Planned

- ⚠️ Checking betting odds 
<br>Tried comparing advantage of past favorite/underdog status in fights, that lead to a ~-0.5% decrease in accuracy, have to try a different approach.
- Categorizing fighters and both checking IF they have fought against someone in that category, and if they did, how they competed
- Octagon time for fighters
- Improve the model to include more in the prediction (Predicted time, Outcome of DEC/SUB/KO)

## Contributing <a name = "contributing"></a>

Feel free to fork this repository, make changes, and submit a pull request.

## 📧 Contact <a name = "contact"></a>

For any inquiries, feel free to reach out:

Email: [markbakosss@gmail.com](mailto:markbakosss@gmail.com) <br>
GitHub: [markbakos](https://github.com/markbakos)
