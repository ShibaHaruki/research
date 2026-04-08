# configs/experiment_sets.py
COMMON_SETS = {
    "base": {
        "VELOCITY" : [20,30,40,50,60,70,80,90,100],

        "BASE_SEED": 12345,
        "NUM_REPEAT": [5],
        "dt_ms": 0.1,

        "SLICE_START": [3000],
        "SLICE_END": [8000],

        "NUM_SAMPLE": 324,
    },

}

TRAINING_SETS = {
    "base": {
        "TRAINING_MAT": ["Al_board","buta_omote","buta_ura","cork","denim","rubber_board","washi","wood_board"],
        "NUM_TRAINING_EPOCH": [3],
        "NUM_TRIANING_SAMPLE": [100],

    },

}

TEST_SETS = {
    "base": {
        "TEST_MAT": ["Al_board","buta_omote","buta_ura","cork","denim","rubber_board","washi","wood_board"],
        "NUM_TEST_SAMPLE": [100],
        "BIN_STEPS": [10],
    },
}





