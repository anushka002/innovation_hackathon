cd ~/
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
sh ./models/download-ggml-model.sh base.en
# build the project
cmake -B build
cmake --build build --config Release

# transcribe an audio file
cd ~/hackathon/src
python3 run.py