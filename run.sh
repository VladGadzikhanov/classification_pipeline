# Flake8
echo 'Checking the code with Flake8:'
flake8 --config=setup.cfg
echo

# Mypy
echo 'Checking the code with mypy:'
mypy --config=setup.cfg .
echo

# Black
echo 'Checking and formating the code with black:'
black --line-length 120 --target-version 'py39' --extend-exclude '''etc/''' --extend-exclude '''src/models''' .
echo

# Isort
echo 'Checking and formating the code with isort:'
isort --profile black --line-length 120 --skip etc --skip src/models .

