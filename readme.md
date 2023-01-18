### Create virtual environment (Windows only)
+ `python -m venv venv`
+ `venv/Scripts/activate`
+ You are now in the activated venv
+ To deactivate your venv type `(venv) deactivate()`

### Jupyter create Kernel
+ `(venv) pip install ipykernel`
+ `(venv) python -m ipykernel install --user --name=<kernel-name>`
+ `(venv) jupyter kernelspec list` to show all available kernels

### Run Tests
+ `pytest hmm/`

### Project Structure
+ This project uses asolute imports 
+ If you want to run a script that is not in the base directory you have to envoke it as a module
+ e.g. the main method of `/hmm/bw.py` can be executed with `python -m hmm.bw`

### Octave
+ Issues: Octave handles size differently than numpy
+ [[1,2,3]].shape != [1,2,3].shape in numpy. In matlab ist das aber schon das selbe
+ warum ist die dimension von B in tau und nu anders?

### Data
+ Important! Data is excluded from git
+ FSDD Recordings have to be placed in `/data/fsdd`

