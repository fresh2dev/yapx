ARG IMAGE_REGISTRY=docker.io
FROM ${IMAGE_REGISTRY}/nginx:1

ARG CONTENT_PATH=public
COPY $CONTENT_PATH /usr/share/nginx/html
