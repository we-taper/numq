cd ..
rm -rf dist/
pip install --upgrade twine wheel
python setup.py sdist bdist_wheel
twine check dist/*
twine upload --repository pypi dist/*
