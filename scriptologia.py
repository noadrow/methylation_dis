import tkinter as tk
from tkinter import filedialog
import importlib.util
import argparse
from collections import defaultdict

script_data = defaultdict()
def load_script():
    file_path = filedialog.askopenfilename(title="Select a Python script", filetypes=(("Python files", "*.py"), ("All files", "*.*")))
    if file_path:
        spec = importlib.util.spec_from_file_location("external_script", file_path)
        external_script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(external_script)

        # Get the arguments for the script from the user
        args_str = arguments_entry.get()

        # Save the script path and arguments for later execution
        script_data["path"] = file_path
        script_data["args"] = args_str

def run_script():
    # Retrieve the script path and arguments
    file_path = script_data.get("path")
    args_str = script_data.get("args")

    if file_path and args_str:
        spec = importlib.util.spec_from_file_location("external_script", file_path)
        external_script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(external_script)

        # Split the arguments by space
        args_list = args_str.split()

        # Call the function from the loaded script with the provided arguments
        #result = external_script.my_function(*args_list)
        external_script.my_function(["only_health_blood_lilifores_GSE87571.txt","only_health_blood_lilifores_GSE87571.bed",1])
        # Display the result on the GUI label
        result_label.config(text=result)

# Create the main application window
app = tk.Tk()
app.title("External Script Loader")

# Create an Entry widget to input arguments
arguments_entry = tk.Entry(app, width=50)
arguments_entry.pack(pady=10)

# Create the Load button
load_button = tk.Button(app, text="Load Script", command=load_script)
load_button.pack(pady=10)

# Create the Run button
run_button = tk.Button(app, text="Run Script", command=run_script)
run_button.pack(pady=10)

# Label to display the result from the loaded script's function
result_label = tk.Label(app, text="", font=("Arial", 12))
result_label.pack(pady=10)

# A dictionary to store script data
script_data = {}

app.mainloop()
