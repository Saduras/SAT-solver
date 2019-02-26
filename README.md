# SAT-solver
UvA 2019 Information Retrieval assignment

## Run with docker
```
# build
docker build -t sat -f ./Dockerfile .

# run
docker run -v <absolut-path-to-data-directory>:/data sat /data sudoku-example-processed.txt
```

## Run with python
```
python sat.py ./data/sudoku-example-processed.txt
```

## Development

Run all unit tests
```
python -m unittest discover -s ./src -p '*_tests.py'
```
