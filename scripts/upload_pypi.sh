cd ..
pip install --upgrade twine
python setup.py sdist bdist_wheel
twine check dist/*
twine upload --repository testpypi dist/*
