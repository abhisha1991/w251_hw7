FROM abhisha1991/fdcustom:fdcustom
USER root

WORKDIR /
RUN mkdir /datacon
# assumes root is the base of the repo
COPY . /datacon
WORKDIR /datacon
