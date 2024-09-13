# see URLs below to create QPM login, download QPM deb package, and sign agreements needed to build this image
# build image with your Qualcomm credentials via "docker build --build-arg QPM_USER=foo --build-arg QPM_PASS=bar -t eigen-hex ."
# run eigen test suite with hexagon simulator via "docker run --rm -it eigen-hex"

FROM ubuntu:22.04
# create a qualcomm account at https://myaccount.qualcomm.com/signup
ARG QPM_USER
ARG QPM_PASS

# install qpm dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends sudo bc libicu70 libsasl2-2 libsqlite3-0 librtmp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# install qpm
# download QPM from https://qpm.qualcomm.com/#/main/tools/details/QPM3
ADD qpm3.deb /
RUN dpkg -i qpm3.deb

# login to qpm and install hexagon sdk 5.x
# sign agreements at https://www.qualcomm.com/agreements
# hexagon installs to /local/mnt/workspace/Qualcomm/...
RUN mkdir -p /local/mnt/workspace
# hexagon installer uses /usr/bin/python
# hexagon installer needs unzip to unpack android ndk
RUN apt-get update && \
    apt-get install -y --no-install-recommends python-is-python3 unzip && \
    rm -rf /var/lib/apt/lists/*
RUN qpm-cli --login $QPM_USER $QPM_PASS && \
    qpm-cli --license-activate hexagonsdk5.x && \
    echo y | qpm-cli --install hexagonsdk5.x && \
    rm -rf /tmp/*

# install hexagon-sim dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libncurses5 libtinfo5 libatomic1 && \
    rm -rf /var/lib/apt/lists/*

# install eigen dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake make && \
    rm -rf /var/lib/apt/lists/*

# clone repo, compile tests
SHELL ["/bin/bash", "-c"]
RUN git clone --filter=blob:none https://gitlab.com/libeigen/eigen.git /eigen && \
    mkdir /build  && \
    cd /build &&\
    source /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.0.1/setup_sdk_env.source && \
    cmake ../eigen -DCMAKE_TOOLCHAIN_FILE=../eigen/cmake/HexagonToolchain.cmake -DBUILD_TESTING=ON && \
    make -j 40 buildtests

WORKDIR /build
CMD ctest -j 40