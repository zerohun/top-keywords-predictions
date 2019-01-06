FROM jupyter/pyspark-notebook

# Install any needed packages specified in requirements.txt
RUN pip install textblob
RUN pip install langdetect

