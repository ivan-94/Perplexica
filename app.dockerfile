FROM node:alpine

WORKDIR /home/perplexica

COPY ui /home/perplexica/

RUN yarn install

ARG NEXT_PUBLIC_WS_URL
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL}
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

RUN yarn build

CMD ["yarn", "start"]