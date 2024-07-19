# deduce_asymptotics

`deduce_asymptotics` is a Python package for deducing the time complexity of functions in big-O notations. It allows you to collect runtime data for a function, fit different time complexity models to the data, and visualize the results with both regular and logarithmic scales.

## Installation

To install the package directly from the GitHub repository, use the following commands:

```bash
git clone https://github.com/mkhlyzov/deduce_asymptotics.git
cd deduce_asymptotics
pip install .
```
Or the editable installation:
```bash
pip install -e .
```

## Usage

After installing the package, you can use it in your Python code as follows:

```python
from deduce_asymptotics import deduce

def build_input(n):
    # Implement your input data generation logic here
    pass

def your_function(input_data):
    # Implement your function to be tested here
    pass

# Call deduce_asymptotics with your input generation and function
deduce(your_function, build_input, time_budget=10, num_samples=10)
# time budget (seconds): Total amount of time the program is allowed to run. More of a soft upper boundary.
# num_samples (int): Number of times function has to be evaluated at each point. Accounts for stochasticity.
```
To customize the rate at which input length gets changed:
```python
deduce(
    your_function, build_input, time_budget=10, num_samples=5,
    step=lambda n: n + 100,
    start=100,
)
```
## Examples

For examples of how to use the package, check out the examples folder in the GitHub repository.
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

If you would like to contribute to this project, please submit a pull request or open an issue on the GitHub repository.
