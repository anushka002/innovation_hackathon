cd ~/

python3 -m venv hack

git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
sh ./models/download-ggml-model.sh base.en
# build the project
cmake -B build
cmake --build build --config Release

source ~/hack/bin/activate

cd ~/hackathon/src

pip3 install -r requirements.txt