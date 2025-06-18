# IA-Project

In this project, an Active Learning model was developed to serve as a medical diagnosis model utilizing a neural network classifier. The model incrementally improves its performance by selecting the most informative samples from the dataset through Active Learning, leveraging the modAL library.

The neural network classifier was implemented using PyTorch and wrapped with Skorch to integrate with the Scikit-learn ecosystem for easier training and evaluation.

This project was conducted in Python 4.2.5 and uses specific versions of the libraries to ensure compatibility and stability.

If you are trying to execute the version .py, we recommend that you put the dataset at the same directory as the code. The following command should be run in a terminal (Anaconda prompt or Python):

	pip install pandas
	pip install numpy
	pip install torch
	pip install skorch
	pip install modAL-python
	pip install matplotlib
	pip install scikit-learn
	pip install tabulate

If library errors persist, it may be necessary to create a new environment in Python(Anaconda).
