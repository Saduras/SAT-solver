# SAT-solver
UvA 2019 Information Retrieval assignment

## Heuristics
1) No Heuristic / Next
2) DLIS_min
3) DLIS_max
4) BOHM
5) Pareto Dominant
6) Random Forest
7) Random

## Run with docker
```
# build
docker build -t sat-solver -f ./Dockerfile .

# run
docker run -v <absolut-path-to-data-directory>:/data sat-solver /data/sudoku-example-processed.txt -S2

# save image
docker save sat-solver -o ./sat-solver

# load image
docker load -i ./sat-solver
```

## Run with python

Requires python +3.6.x. Easiest way to install all pip dependencies is to create an anaconda environment from the `environment.yml` file.
```
# create environment
conda env create -f environment.yml

# activate conda environment
conda activate SAT-solver

# run sat solver
python ./src/sat.py ./data/sudoku-example-processed.txt -S2
```

Two use the Random Forest heuristic you have to download the model from [https://drive.google.com/file/d/1Y_DlmvIdK7AHuEiuIoDszIqCeoaDQUej/view?usp=sharing](https://drive.google.com/file/d/1Y_DlmvIdK7AHuEiuIoDszIqCeoaDQUej/view?usp=sharing) and place it in `<project-root>/models/RFfinalized_model.sav`.
The use heuristic 6 (Random Forest) when calling `sat.py`:
```
python ./src/sat.py ./data/sudoku-example-processed.txt -S6
```


## Development

Run all unit tests
```
python -m unittest discover -s ./src -p '*_tests.py'
```
