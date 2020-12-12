FROM cupy/cupy:v8.2.0

COPY ./src /tmp/src
COPY ./MANIFEST.in /tmp/MANIFEST.in
COPY ./README.md /tmp/README.md
COPY ./requirements.txt /tmp/requirements.txt
COPY ./requirements_tf.txt /tmp/requirements_tf.txt
COPY ./requirements_dev.txt /tmp/requirements_dev.txt
COPY ./setup.cfg /tmp/setup.cfg
COPY ./setup.py /tmp/setup.py
COPY ./versioneer.py /tmp/versioneer.py
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements_dev.txt
RUN pip3 install -r /tmp/requirements_tf.txt
RUN pip3 install -e /tmp/
CMD pytest /tmp/src/numq/tests
