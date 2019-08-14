export PYTHONPATH="/home/ubuntu/NeuralAlgorithmSelection/:$PYTHONPATH"
cd ..
pip install -r requirements.txt
cd aws
pip install torch-cluster
pip install torch-sparse
pip install torch-scatter
pip install torch-geometric
pip install lmdb protobuf==3.7.1
