# ipl_score_prediction
IPL Score Prediction Using ANN

# IPL Score Prediction

This repository contains a project that predicts the total score of an IPL team at the end of their innings using an Artificial Neural Network (ANN). The dataset used for this project consists of ball-by-ball details of IPL matches.

## Dataset

The dataset contains 76014 rows and 15 columns. Below is the description of each column:

- **mid**: Unique match id.
- **date**: Date on which the match was played.
- **venue**: Stadium where the match was played.
- **bat_team**: Batting team name.
- **bowl_team**: Bowling team name.
- **batsman**: Batsman who faced that particular ball.
- **bowler**: Bowler who bowled that particular ball.
- **runs**: Runs scored by the team till that point of instance.
- **wickets**: Number of wickets fallen of the team till that point of instance.
- **overs**: Number of overs bowled till that point of instance.
- **runs_last_5**: Runs scored in previous 5 overs.
- **wickets_last_5**: Number of wickets that fell in previous 5 overs.
- **striker**: Max(runs scored by striker, runs scored by non-striker).
- **non-striker**: Min(runs scored by striker, runs scored by non-striker).
- **total**: Total runs scored by the batting team at the end of the first innings.

## Project Workflow

1. **Data Exploration and Preprocessing**:
    - Loaded the dataset and checked its basic information (shape, description, info).
    - Checked for null values using `df.isnull().sum()`.
    - Dropped the "mid" column as it was all null and the "date" column as it was not necessary for the prediction.
    - Performed label encoding on categorical columns (`venue`, `bat_team`, `bowl_team`, `batsman`, `bowler`) using `LabelEncoder` from `sklearn`.
    
    ```python
        df['venue'] = venue_encoder.fit_transform(df["venue"])
        df['bat_team'] = bat_team_encoder.fit_transform(df["bat_team"])
        df['bowl_team'] = bowl_team_encoder.fit_transform(df["bowl_team"])
        df['batsman'] = batsman_encoder.fit_transform(df["batsman"])
        df['bowler'] = bowler_encoder.fit_transform(df["bowler"])
    ```
    - Analyzed the correlation between the columns and plotted a heatmap to visualize it.
    ```python
        df.corr()["total"].sort_values()
        sns.heatmap(df.corr(), cmap="viridis")
    ```
    - Plotted a boxplot to see the distribution of the data.

2. **Model Training**:
    - Split the dataset into training and testing sets using `train_test_split` from `sklearn`.
    - Scaled the data using `MinMaxScaler` from `sklearn` to improve the performance of the neural network.
    - Defined a Sequential model from `tf.keras.models` and added Dense and Dropout layers from `tf.keras.layers`.
    - Trained multiple models with different dense and dropout layers to find the best combination.
    - Settled on the final model after experimenting with different configurations.
    ```python
        model = Sequential()

        model.add(InputLayer(input_shape=(12,)))

        model.add(Dense(1024, activation='relu'))

        model.add(Dense(512, activation='relu'))

        model.add(Dense(256, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))

        model.add(Dense(1, activation="linear"))

        model.compile(optimizer='adam', loss="mean_squared_error")

    ```
    - Trained the model using the training data with 100 epochs.
    ```python
        history = model.fit(x= X_train,
         y = y_train,
         epochs = 100,
         validation_data=[X_test, y_test])
    ```

3. **Model Evaluation**:
    - Evaluated the model on the test set and received a loss of 78.4012.
    - Plotted the loss vs validation loss.
    - Calculated the mean squared error (MSE), root mean squared error (RMSE), and r2 score:
        - **MSE**: 78.54
        - **RMSE**: 8.86
        - **R2 Score**: 0.906 (which is very good)

## Results

- The final model achieved a very good r2 score of 0.906, indicating that it can accurately predict the total score of an IPL team at the end of their innings.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/riitk/ipl_score_predictions.git
    cd ipl_score_predictions
    ```

## Conclusion

This project demonstrates the use of an ANN for predicting IPL scores. The high r2 score indicates the model's effectiveness in making accurate predictions. Further improvements can be made by experimenting with other neural network architectures and tuning hyperparameters.

## Acknowledgements

- The dataset was sourced from Kaggle: [IPL Data](https://www.kaggle.com/datasets/yuvrajdagur/ipl-dataset-season-2008-to-2017).
