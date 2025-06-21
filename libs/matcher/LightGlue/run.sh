# create a venv
python -m venv ./venv
source venv/bin/activate

# setup and install the lightglue from the current source
python -m pip install -e .


# run the matching demo
python ./demo_matching_lightglue.py



