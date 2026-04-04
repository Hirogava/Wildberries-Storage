FROM golang:1.26 AS build

WORKDIR /src

COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal

RUN go build -o /out/wb-storage-api ./cmd/server

FROM debian:bookworm-slim

WORKDIR /app

ENV APP_ADDR=:8080

COPY --from=build /out/wb-storage-api /usr/local/bin/wb-storage-api
COPY .env.example ./
COPY README.md ./
COPY docs ./docs
COPY train_team_track.parquet ./train_team_track.parquet
COPY test_team_track.parquet ./test_team_track.parquet

EXPOSE 8080

CMD ["wb-storage-api"]
