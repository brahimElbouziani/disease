WORKING_DIR="tf_files"

BOTTLENECK_DIR="$WORKING_DIR/Pepper/"
STEPS=10000
MODEL_DIR="$WORKING_DIR/inception"
OUTPUT_GRAPH="$WORKING_DIR/retrained_graph_Pepper.pb"
OUTPUT_LABELS="$WORKING_DIR/retrained_labels_Pepper.txt"
DATA_FOLDER="$WORKING_DIR/data/Pepper/"

python3 train.py \
--bottleneck_dir=$BOTTLENECK_DIR \
--how_many_training_steps $STEPS \
--model_dir=$MODEL_DIR \
--output_graph=$OUTPUT_GRAPH \
--output_labels=$OUTPUT_LABELS \
--image_dir $DATA_FOLDER
