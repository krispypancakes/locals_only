# localmistral
Github's suggested name for this repo is 'reimagined-guacamole' and I like it.

## goal
Run mistral 7b locally on Apple M1 Pro, 32GB

## installation

`CMAKE_ARGS="-DLLAMA_METAL=on, -DCMAKE_APPLE_SILICON_PROCESSOR=arm64" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir` to properly install llamacpp

