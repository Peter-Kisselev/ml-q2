# ml-q2
Machine Learning Quarter 2 project

A project designed to improve random forests by choosing their attributes better.
The method, which we call Dynamic Semi-Random Forests (DSRFs), creates a subset of decision forests with samples of the attributes.
It then evaluates the subsets on a validation dataset, and assigns each attribute a score correlary to its performance.
It repeats this until eventually settling on a best set of attributes, at that point returning a final random forest constructed with those attributes.

This project was co-authored by Peter Kisslev, Nikhil Alladi, and Jacob Dipasupil.
