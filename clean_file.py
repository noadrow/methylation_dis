from tkinter.filedialog import askopenfilename
def process_line(line):
    # Split the line by semicolon
    parts = line.split(';')
    # Return the unique parts as a string
    return ';'.join(set(parts))

def main(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            processed_line = process_line(line.strip())
            outfile.write(processed_line + '\n')

if __name__ == "__main__":
    input_file_path = askopenfilename()
    output_file_path = str(input_file_path.split(".")[0])+"_clean.txt"
    main(input_file_path, output_file_path)
    print("Processing complete.")


