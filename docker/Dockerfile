FROM cupy/cupy:v8.2.0

COPY ./ /tmp
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements_dev.txt
RUN pip3 install -r /tmp/requirements_tf.txt
RUN pip3 install -e /tmp/
ENTRYPOINT pytest /tmp/src/numq/tests
