#!/usr/bin/env python



# given a model path return its step number
def get_global_step(model_path):
    return int(model_path.split('-')[-1])