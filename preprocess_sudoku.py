import sys;

def append_rules(filename, rule_path, output):
    """
    Concatenate DIMACS sudoku with DIMACS sudoku rules to signale file
    input:
        filename - path to sudoku in DIMACS format
        rule_path - path to sudoku rule set in DIMACS format
        output - path to store result
    """
    with open(rule_path, 'r') as file:
        rules = file.read()

    with open(filename, 'r') as file:
        sudoku = file.read()

    with open(output,'w') as file:
        file.write(rules + sudoku)
    print('Created combined file at ' + output)

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print("Missing arguments! Expecting 'sudoku-filename' 'output-path'")
        exit()

    rule_path = './data/sudoku-rules.txt'
    filename = sys.argv[1]
    output = sys.argv[2]

    append_rules(filename, output)