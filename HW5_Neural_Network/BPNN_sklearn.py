'''
Chaoyu Li
Date: 3/30/2022
'''
from sklearn.neural_network import MLPClassifier

def read_PGM_img(pic):
    with open(pic, 'rb') as file:
        file.readline()
        file.readline()
        dimList = file.readline().split()
        dimX = int(dimList[0])
        dimY = int(dimList[1])
        size = dimX*dimY

        max_greyscale = int(file.readline().strip())

        image = []
        for i in range(size):
            pixel = file.read(1)[0]
            image.append(pixel/max_greyscale)
        return image

def main(train_List_Dir, test_List_Dir):
    # initialize lists of images and labels
    images = []
    labels = []

    # import training data
    with open(train_List_Dir) as f:
        for line in f.readlines():
            # get file directory
            train_img_dir = line.strip()

            # import the images
            images.append(read_PGM_img(train_img_dir))

            # assign label based on file name: 1 for down gesture, 0 for otherwise
            if 'down' not in train_img_dir:
                labels.append(0)
            else:
                labels.append(1)

    nn = MLPClassifier(solver='sgd', tol=1e-3, alpha=0.1, learning_rate='adaptive',
                      hidden_layer_sizes=(100,), activation='logistic', learning_rate_init=0.1,
                      max_iter=1000, verbose=False, warm_start=True, early_stopping=False, validation_fraction=0.1)

    nn.fit(images, labels)

    f.close()

    # predict testing data
    total_count = 0
    correct_count = 0
    with open(test_List_Dir) as f:
        for line in f.readlines():
            total_count += 1
            test_image = line.strip()

            pred = nn.predict([read_PGM_img(test_image), ])[0]
            if (pred == 1) == ('down' in test_image):
                correct_count += 1

    print('correct_count rate on test data: {}'.format(correct_count / total_count))

if __name__ == '__main__':
    main('downgesture_train.list', 'downgesture_test.list')