FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG TRT_VERSION=10.9.0.34-1+cuda12.8
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg \
    && curl -fL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer10=${TRT_VERSION} \
    libnvinfer-plugin10=${TRT_VERSION} \
    libnvinfer-lean10=${TRT_VERSION} \
    libnvinfer-vc-plugin10=${TRT_VERSION} \
    libnvinfer-dispatch10=${TRT_VERSION} \
    libnvonnxparsers10=${TRT_VERSION} \
    libnvinfer-bin=${TRT_VERSION} \
    && rm -f /tmp/cuda-keyring.deb \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/src/tensorrt/bin:${PATH}
WORKDIR /work
COPY smoke-test.sh /usr/local/bin/foundationpose-smoke-test
RUN chmod +x /usr/local/bin/foundationpose-smoke-test
CMD ["foundationpose-smoke-test"]
