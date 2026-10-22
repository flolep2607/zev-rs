# zev-router: build the router crate alone (it is its own workspace), ship the binary on a slim base.
#   docker buildx build -f router/Dockerfile -t <registry>/zev-router:<git sha> router/
FROM rust:1-bookworm AS build
WORKDIR /src
# libz-ng-sys (flate2's zlib-ng backend) builds with cmake.
RUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/*
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin inference_router_rs \
    && cp target/release/inference_router_rs /usr/local/bin/zev-router

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/zev-router /usr/local/bin/zev-router
ENTRYPOINT ["zev-router"]
CMD ["/etc/zev-router/router.json"]
