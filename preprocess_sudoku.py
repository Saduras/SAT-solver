import sys;

if(len(sys.argv) != 3):
    print("Missing arguments! Expecting 'sudoku-filename' 'output-path'")
    exit()

rule_path = './data/sudoku-rules.txt'
filename = sys.argv[1]
output = sys.argv[2]

with open(rule_path, 'r') as file:
    rules = file.read()

with open(filename, 'r') as file:
    sudoku = file.read()

with open(output,'w') as file:
    file.write(rules + sudoku)

print('Created combined file at ' + output)

