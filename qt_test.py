import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtCore import Qt
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
import numpy as np
import matplotlib.pyplot as plt

Tk().withdraw()

print('waiting for data')
print('...')
INPUT_control = askopenfilename(title='Insert csv table (methylation data)')
print('loading data')
print('...')
t0 = time.time()
df = pd.read_csv(INPUT_control,index_col=0)
print(f'data loaded in {time.time()-t0} seconds')
print('...')

def barplot_range_count(df, group, cgs):
    # cgs = df.index.to_list()[:len(df) - 4]

    for cg in cgs:
        if (cg in df.index):
            working_df = df.loc[cg]
            new_pd = pd.DataFrame({
                'range': pd.cut(working_df, np.arange(0, 1, 0.01)),
                'val': working_df,
                'index': working_df.index,
                'counter': [1] * len(working_df)
            })

            range_count = new_pd.groupby('range')['counter'].count()

            range_count.plot.bar(rot=90, figsize=(15, 10))
            #fig = ax.figure
            #fig.savefig(f"{cg}_{group}_barh.png")
            plt.show()
            #plt.clf()
            #plt.close(fig)
        else:
            print("CpG not found")


class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.dataList = []
        self.currentIndex = 0

        self.layout = QVBoxLayout(self)

        self.textEdit = QTextEdit(self)
        self.layout.addWidget(self.textEdit)

        self.loadButton = QPushButton("Load", self)
        self.layout.addWidget(self.loadButton)

        self.graphView = QLabel(self)
        self.layout.addWidget(self.graphView)

        self.tpButton = QPushButton("TP", self)
        self.tnButton = QPushButton("TN", self)
        self.prevButton = QPushButton("Previous", self)
        self.nextButton = QPushButton("Next", self)
        self.layout.addWidget(self.tpButton)
        self.layout.addWidget(self.tnButton)
        self.layout.addWidget(self.prevButton)
        self.layout.addWidget(self.nextButton)

        self.loadButton.clicked.connect(self.load_data)
        self.tpButton.clicked.connect(lambda: self.label_graph("TP"))
        self.tnButton.clicked.connect(lambda: self.label_graph("TN"))
        self.prevButton.clicked.connect(self.show_previous_graph)
        self.nextButton.clicked.connect(self.show_next_graph)

        self.setLayout(self.layout)

    def load_data(self):
        input_data = self.textEdit.toPlainText()
        self.dataList = input_data.split('\n')
        self.currentIndex = 0
        if self.dataList:
            self.show_graph()
    def label_graph(self, label):
        if self.dataList:
            current_item = self.dataList[self.currentIndex]
            print(f"Labeled '{current_item}' as '{label}'")

    def show_previous_graph(self):
        if self.dataList:
            self.currentIndex = (self.currentIndex - 1 + len(self.dataList)) % len(self.dataList)
            self.show_graph()

    def show_next_graph(self):
        if self.dataList:
            self.currentIndex = (self.currentIndex + 1) % len(self.dataList)
            self.show_graph()

    def show_graph(self):
        current_item = self.dataList[self.currentIndex]
        self.plot_graph(current_item)
        barplot_range_count(df, "chosen_cgs", [current_item])
        self.graphView.setText(current_item)

    def plot_graph(self, data):
        # Replace this function with your actual graph plotting logic
        # For the sake of example, we're just printing the data
        print("Plotting graph for:", data)

def main():
    app = QApplication(sys.argv)

    widget = GraphWidget()
    widget.setGeometry(100, 100, 800, 600)
    widget.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()