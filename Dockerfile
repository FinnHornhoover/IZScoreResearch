# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update
RUN apt-get install -y wget

# Install pip requirements
RUN wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.1-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.9.1-linux-x86_64.tar.gz --directory /opt
RUN rm julia-1.9.1-linux-x86_64.tar.gz
ENV PATH="${PATH}:/opt/julia-1.9.1/bin"

RUN python3 -m pip install --upgrade pip
RUN pip3 install ipython
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install matplotlib
RUN pip3 install torch
RUN pip3 install pysr
RUN python3 -m pysr install

WORKDIR /app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python3", "score_search.py", "--fit-type", "per_iz"]
