# Soccer Player Rating Prediction

## Installation Guide
1. Clone or Fork the Project
2. Create a Virtual Enviroment
3. go to same virtual enviroment and write below cmd
4. pip install -r requirements.txt


### 1. Project Description
#### A. Problem Statement

The dataset you are going to use is from European Soccer Database (https://www.kaggle.com/hugomathien/soccer) has more than 25,000 matches and more than 10,000 players for European professional soccer seasons from 2008 to 2016. We need to predict the Soccer player's rating based on some of his features.<br>
The European Soccer Database on Kaggle provides a comprehensive dataset of European soccer matches, players, and teams from 2008 to 2016. The dataset includes tables such as:

1. Match: Information about individual matches (e.g., date, team names, scores).
2. Player: Data about players (e.g., name, birth date, height, weight).
3. Player_Attributes: Player performance metrics (e.g., overall rating, potential, preferred foot).
4. Team: Details about teams (e.g., team name, team FIFA API ID).
5. Team_Attributes: Team performance metrics (e.g., build-up play, chance creation).<br>
We are going to use Team_attribute teble for our task. It contains columns as ->
player_fifa_api_id: Unique identifier for the player in the FIFA database.<br>
player_api_id: Unique identifier for the player.<br>
potential: The player's potential rating.<br>
crossing: Ability to deliver accurate crosses.<br>
finishing: Ability to finish scoring chances.<br>
heading_accuracy: Accuracy of headers.<br>
short_passing: Ability to make short passes accurately.<br>
volleys: Ability to volley the ball.<br>
dribbling: Ability to dribble past opponents.<br>
curve: Ability to curve the ball.<br>
free_kick_accuracy: Accuracy of free kicks.<br>
long_passing: Ability to make long passes accurately.<br>
ball_control: Ability to control the ball.<br>
acceleration: Speed of initial movement.<br>
sprint_speed: Maximum running speed.<br>
agility: Ability to move quickly and change direction.<br>
reactions: Ability to react quickly to situations.<br>
balance: Ability to maintain balance.<br>
shot_power: Power of shots.<br>
jumping: Ability to jump high.<br>
stamina: Endurance level.<br>
strength: Physical strength.<br>
long_shots: Ability to score from long distances.<br>
aggression: Level of aggressiveness.<br>
interceptions: Ability to intercept passes.<br>
positioning: Ability to position oneself effectively.<br>
vision: Ability to see and make accurate passes.<br>
penalties: Ability to take penalty shots.<br>
marking: Ability to mark opponents.<br>
standing_tackle: Ability to perform standing tackles.<br>
sliding_tackle: Ability to perform sliding tackles.<br>
gk_diving: Goalkeeper's diving ability.<br>
gk_handling: Goalkeeper's handling ability.<br>
gk_kicking: Goalkeeper's kicking ability.<br>
gk_positioning: Goalkeeper's positioning ability.<br>
gk_reflexes: Goalkeeper's reflexes.<br>
`Three categorical columns are:`<br>
preferred_foot: The player's preferred foot (left or right).<br>
attacking_work_rate: The player's work rate in attacking (e.g., low, medium, high).<br>
defensive_work_rate: The player's work rate in defending (e.g., low, medium, high).<br>

#### B. Tools and Libraries
Tools<br><br>
a.Python<br>
b.SQL<br>
c.Jupyter Notebook<br>
d. Flask<br>
e. HTML<br>
f. Render<br>
g. GitHub

Libraries<br><br>
a.Pandas<br>
b.Scikit Learn<br>
c.Numpy<br>
d.Seaborn<br>
e.Matpoltlib<br>
f.sqlite3

### 2. Data Collection
The dataset comes in the form of an SQL database and contains statistics of about 25,000 football matches, from the top football league of 11 European Countries. It covers seasons from 2008 to 2016 and contains match statistics (i.e: scores, corners, fouls etc...) as well as the team formations, with player names and a pair of coordinates to indicate their position on the pitch.<br>
There are 42 columns and 183978 Rows. These are the major point about the data set.<br>


Target Column is Rating which can range from 0-100

### 3. EDA
#### A.Data Cleaning
There is no need of three columns such as id, player_fifa_api_id, player_api_id and date, so those columns can be dropped<br>
Now We have 38 columns and null values are present, so null values should be handled<br>
Null values for numerical columns can be handled via median stargegy and for categorical columns mode can be applied<br>
Dependent Variable is overall_rating

#### B. Feature Engineering
No outliers are present in the data

#### C. Data Normalization
Normalization (min-max Normalization)<br>
In this approach we scale down the feature in between 0 to 1<br>
There are different ways by which we can convert categorical cols to numerical cols such as ->
1. Label Encoding -> Label Encoding converts categorical values to numerical values by assigning a unique integer to each category. This is useful for ordinal categorical variables.
2. One-Hot Encoding -> One-Hot Encoding creates binary columns for each category. This is useful for nominal categorical variables.
3. Ordinal Encoding -> Ordinal Encoding is used when the categorical variable has a natural order but no fixed spacing between the categories.
4. Frequency Encoding -> Frequency Encoding replaces categories with their frequency in the dataset.
5. Target Encoding -> Target Encoding replaces categories with the mean of the target variable for each category. This is more advanced and should be used carefully to avoid data leakage.

We use ordinal encoding to convert categorical cols to numerical cols in our use-case

We have numerical column where we can apply min-max Normalization.<br>

### 4. Choosing Best ML Model
List of the model that we can use for our problem<br>
a. LinearRegression<br>
b. DecisionTree<br>
c. RandomForestRegressor<br>
d. XGBRegressor<br>

### 5. Model Creation
From the above 4 Models, we chosse the model where we are having best r2 score.

Algorithm that can be used for Hyperparameter tuning are :-

a. GridSearchCV<br>
b. RandomizedSearchCV<br>

Different parameters based on different models are used for hyper-parameter tuning, some of the key paramters are :-

a. n_estimators -> Number of trees in the forest.<br>
b. criterion -> 'mse' (mean squared error, default) or 'mae' (mean absolute error).<br>
c. max_depth -> The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain fewer than min_samples_split samples.<br>
d. min_samples_split -> The minimum number of samples required to split an internal node. Default is 2.<br>
e. min_samples_leaf -> The minimum number of samples required to be at a leaf node. Default is 1.<br>
f. max_leaf_nodes -> Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None (default), then unlimited number of leaf nodes.<br>
g. max_features -> The number of features to consider when looking for the best split. `None` (default): Consider all features.<br>
h. alpha ---> Alpha is a positive constant that multiplies the regularization terms. It controls the overall strength of regularization applied to the model.<br>
i. l1_ratio ---> l1_ratio is the mixing parameter that controls the balance between L1 (Lasso) and L2 (Ridge) penalties in Elastic Net regularization.<br>

We achieved the below data as final O/P ->
`{'LinearRegression': {'best_params': {}, 'test_r2_score': 0.8430755734126719}, 'DecisionTree': {'best_params': {'max_depth': 15}, 'test_r2_score': 0.945131965458793}, 'RandomForestRegressor': {'best_params': {'max_depth': 12, 'max_features': 'sqrt', 'n_estimators': 100}, 'test_r2_score': 0.9579811223552948}, 'XGBRegressor': {'best_params': {'learning_rate': 0.3, 'max_depth': 9, 'n_estimators': 100}, 'test_r2_score': 0.9801681873519132}}`

### 6. Model Deployment
After creating model ,we integrate that model with beautiful UI. for the UI part we used HTML and Flask. We have added extra validation check also so that user doesn't enter Incorrect data. Then the model is deployed on render

### 7. Model Conclusion

Model predict 0.98 accurately on test data(R2 Score).

### 8. Project Innovation
a. Easy to use<br>
b. Open source<br>
c. Best accuracy<br>
d. GUI Based Application

### 9. Limitation And Next Step
Limitation are :-<br>
a. Mobile Application<br>
b. Accuracy can be improved more<br>
d. Feature is limited

Next Step are :-<br>
a. we can work on mobile application<br>

## Deployable Link
https://machine-learning-practical-04-boston.onrender.com/predict