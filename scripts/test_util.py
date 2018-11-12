import re


def train():
    scope = 'conv_1'
    c = []
    collection = ['conv_1/weights', 'conv_11/weights', 'conv_12/weights', 'conv_13/weights']
    regex = re.compile(scope)
    for item in collection:
        if regex.match(item):
            print item

if __name__ == '__main__':
    train()