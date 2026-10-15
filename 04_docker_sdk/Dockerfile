# Image sdk_pynqz2 : SDK PetaLinux (compilation croisee ARM) pour LoCoD.
# Construction : placer sdk-pynqz2.sh (fourni par Viveris) a cote de ce fichier, puis
#   docker build -t sdk_pynqz2:latest .
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 xz-utils gcc g++ make file ca-certificates locales \
    && locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
COPY sdk-pynqz2.sh /tmp/sdk-pynqz2.sh
RUN chmod +x /tmp/sdk-pynqz2.sh \
    && /tmp/sdk-pynqz2.sh -y -d /opt/petalinux-sdk \
    && rm /tmp/sdk-pynqz2.sh
WORKDIR /workdir
