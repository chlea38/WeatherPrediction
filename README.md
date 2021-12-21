<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/chlea38/WeatherPrediction">
    <img src="project_logo.png" alt="Logo" width="200" height="200">
  </a>

<h1 align="center">WeatherPrediction</h1>

  <p align="center">
    Miniproject BIO-322: will it rain tomorrow in Pully ?
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About The Project</a>
      <ul>
        <li><a href="#description">Project description</a></li>
        <li><a href="#organization">Repository organization</a></li>
      </ul>
    </li>
    <li>
      <a href="#start">Getting Started</a>
      <ul>
        <li><a href="#prerequisite">Prerequisites</a></li>
        <li><a href="#install">Installation</a></li>
      </ul>
    </li>
    <li><a href="#use">Usage</a></li>
    <li><a href="#contacts">Contributors</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
<a name="about"></a>
## About The Project

<a name="description"></a>
### Project description
The project aims to predict whether or not it will rain in Pully on the next day given some measurements at different weather stations. _(Please refer to the [Kaggle competition](https://www.kaggle.com/c/bio322-will-it-rain-tomorrow/))_

The training and test [data](https://github.com/chlea38/WeatherPrediction/tree/main/data) are CSV files containing measurements of 6 quantity for 22 stations at 4 times (528 columns). The [training data](https://github.com/chlea38/WeatherPrediction/blob/main/data/trainingdata.csv) contains in addition a column filled with Bool, telling whether or not it has been raining in Pully the day after the measurements. The goal of the project is to predict this for the [test data](https://github.com/chlea38/WeatherPrediction/blob/main/data/testdata.csv) using Machine Learning methods and save the prediction in a CSV file.

To this end, different machine learning methods were tested, including multiple logistic regression and random forest classifier. To improve the results, some complementary methods were used, like hyperparameters selftuning, L2 regularization or standardization. Please refer to the report for full description.


<a name="organization"></a>
### Repository organization
The repository contains 3 folders:

* [src](https://github.com/chlea38/WeatherPrediction/tree/main/src) : contains the julia scripts performing weather prediction using different methods.
* [data](https://github.com/chlea38/WeatherPrediction/tree/main/data) : contains the data provided for the project, including the training data, the test data and a submission example.
* [results](https://github.com/chlea38/WeatherPrediction/tree/main/results) : contains the various results files that we produced during the project, with the different methods. 

The repository contains also the report and this README.

<p align="right"><a href="#top">back to top</a></p>


<!-- GETTING STARTED -->
<a name="start"></a>
## Getting Started

<a name="prerequisite"></a>
### Prerequisites

To run the program, you will need these components:
* Julia _(Please refer to the [Julia website](https://julialang.org/downloads/))_ 
* MLCourse environment _(see the [MLCourse repository](https://github.com/jbrea/MLCourse))_

<a name="install"></a>
### Installation
To be able to use the program, you need to clone the repository using : 
   ```sh
   git clone https://github.com/chlea38/WeatherPrediction.git
   ```

<p align="right"><a href="#top">back to top</a></p>


<!-- USAGE -->
<a name="use"></a>
## Usage
The results files can be reproduced easily by running the different scripts. 
1. Open the terminal at WeatherPrediction/src 
2. Type julia and the name of the file you want to run (for example [XGBoost classification](https://github.com/chlea38/WeatherPrediction/blob/main/src/XGBoost.jl)) 
  ```sh
   > julia XGBoost.jl
   ```
3. The output files will be saved in the folder `results`.

_For full results, please refer to the [results](https://github.com/chlea38/WeatherPrediction/tree/main/results) or to the report_

<p align="right"><a href="#top">back to top</a></p>


<!-- CONTACT -->
<a name="contacts"></a>
## Contributors

Chléa Schiff - chlea.schiff@epfl.ch

Méline Cretegny - meline.cretegny@epfl.ch

Project Link: [WeatherPrediction](https://github.com/chlea38/WeatherPrediction)

<p align="right"><a href="#top">back to top</a></p>




