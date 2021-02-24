from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, validation_curve
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io
import time
import cv2
import os


# <editor-fold desc="Data preprocessing">
def GetDefaultParameters():
    '''
        initiating the pipe parameters dictionary
    '''

    path = 'C:/Users/idofi/OneDrive/Documents/BGU\masters/year A/computer vision/Task_2/FlowerData'
    test_images_indices = list(range(301, 473))
    image_size = 224
    threshold = 0.5
    dataAugment = 1
    batch_size = 32
    epochs = 3
    SGD_learning_rate = 0.5
    SGD_momentum = 0.6
    SGD_nesterov = False
    SGD_decay = 1e-6
    RF_max_depth = 7
    RF_n_estimators = 19

    params = {'Data': {'path': path, 'test_images_indices': test_images_indices},
              'Prepare': {'size': image_size, 'dataAugment': dataAugment},
              'Model': {'batch_size': batch_size, 'epochs': epochs, 'threshold': threshold},
              'Optimizer': {'SGD': {'learning_rate': SGD_learning_rate, 'momentum': SGD_momentum,
                                    'nesterov': SGD_nesterov, 'decay': SGD_decay}},
              'RF': {'max_depth': RF_max_depth, 'n_estimators': RF_n_estimators}}

    return params


def GetData(Params, saveToPkl):
    '''
    creating dataframe with the images & labels data
    :param  Params: for the folder path
    :param  saveToPkl: serializing to pickle file (boolean)
    :return dataframe
    '''

    folder_path = Params['Data']['path']

    print(' ---- Importing data ---- ')
    mat_file = scipy.io.loadmat('{}/FlowerDataLabels.mat'.format(folder_path))
    labels = mat_file['Labels'][0]
    raw_data = pd.DataFrame(columns=['Img_ID', 'Data', 'Labels'])

    # read images from folder
    folder_files = os.listdir(folder_path)
    for file in folder_files:
        if file.endswith(('.jpeg')):
            Img_ID = int(file.replace('.jpeg', ''))
            readed_image = cv2.imread('{}\{}'.format(folder_path, file))
            image_label = labels[Img_ID - 1]
            raw_data = raw_data.append({'Img_ID': Img_ID, 'Data': readed_image, 'Labels': image_label},
                                       ignore_index=True)

    # sort daraframe by img_ID
    sorted_raw_data = raw_data.sort_values(['Img_ID'])
    sorted_raw_data = sorted_raw_data.reset_index(drop=True)

    if saveToPkl:
        sorted_raw_data.to_pickle('raw_data.pkl')

    return sorted_raw_data


def TrainTestSplit(Params, DandL):
    '''
    splits the data into train and test
    :param DandL: raw dataframe
    :param Params
    :return: TrainData & TestData (in dataframe format)
    '''

    test_images_indices = Params['Data']['test_images_indices']

    TestData = DandL[DandL['Img_ID'].isin(test_images_indices)]
    TrainData = DandL[~DandL['Img_ID'].isin(test_images_indices)]

    print('Data Split:\nTrain: {} \nTest: {}'.format(TrainData.shape[0], TestData.shape[0]))

    return TrainData, TestData


def prepare(Params, data):
    '''
       resizing the images to fit the resNet50V2
       performig preprocessing and fit it into numpy array
       :param:data: image + labels dataframe
       :param: Params: pipe parameters
       :return: dataframe with resized images
       '''

    copy_data = data.copy()
    size = Params['Prepare']['size']
    # resizing the images
    copy_data['Data'] = copy_data['Data'].apply(lambda x: cv2.resize(x, (size, size)))

    # applying resNetV2 pre_processing to fit the model
    copy_data['Data'] = copy_data['Data'].apply(lambda x: preprocess_input(x))

    train_Set = []
    labels_set = []

    for img in copy_data['Data']:
        train_Set.append(img)

    for label in copy_data['Labels']:
        labels_set.append(label)

    train_Set = np.array(train_Set)
    labels_set = np.array(labels_set)

    return train_Set, labels_set


def DataAugment(TrainData_x, TrainData_y, Amount):
    '''
    creates more images out of the existing data by performing different manipulations
    :param TrainData_x: Train imgs
    :param TrainData_y: Train Lables
    :param Amount: Amount of photos to generate
    :return: Return 2 vectores: one with augmented photos and one with thier Label
    '''

    # print(TrainData_y)
    copy_data = TrainData_x.copy()

    X_with_augment = []
    Y_with_augment = []

    # print('------ Augmenting Data ------')
    # print('Data Shape before augment: ', copy_data.shape)

    ##Declaring Generator params
    train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.1, rotation_range=10
                                       , horizontal_flip=True, fill_mode='nearest')
    for i in range(0, len(copy_data)):

        ##Iterate on each image
        Train_flower_generator = train_datagen.flow(copy_data[i:i + 1], TrainData_y[i:i + 1])

        flo = [next(Train_flower_generator) for i in range(0, Amount)]
        for j in range(0, Amount):
            X_with_augment.append(flo[j][0][0])
            Y_with_augment.append(TrainData_y[i:i + 1][0])

    X_with_augment = np.array(X_with_augment)
    Y_with_augment = np.array(Y_with_augment)

    # print(Y_with_augment)

    print('------ Augmenting Data ------')
    print('Data Shape After augment: ', X_with_augment.shape)

    return X_with_augment, Y_with_augment


# </editor-fold>

# <editor-fold desc="Baseline model">
def build_baseline_model(Params):
    '''
    bulidng a pre-trined baseline model
    :param Params: pipe parameters
    :return: pre=trained resNet model
    '''

    # using the built-in option of avg pooling of the last cov layer
    model_1 = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

    # Taking the output of the ResNet50 vector
    last_layer = model_1.output

    # adding the output layer using the sigmoid function to get probability
    predictions = Dense(1, activation='sigmoid')(last_layer)

    # Model to be trained
    model = Model(inputs=model_1.input, outputs=predictions)

    # Train only the layers which we have added at the end
    for layer in model_1.layers:
        layer.trainable = False

    optimizer = SGD(learning_rate=Params['Optimizer']['SGD']['learning_rate'],
                    momentum=Params['Optimizer']['SGD']['momentum'], nesterov=Params['Optimizer']['SGD']['nesterov'])

    # using SGD(stochastic gradient descent)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model


def train(model, train_x, train_y, Params):
    '''
      train a given model
      :param train_set
      :param: Params: pipe parameters
      :return: trained model
    '''

    model.fit(train_x, train_y, batch_size=Params['Model']['batch_size'], epochs=Params['Model']['epochs'])

    return model


# </editor-fold>

# <editor-fold desc="2layer model">
def build_2layer_model(Params):
    '''
    Build a complex model with one more traind layer
    :param Params: pipe parameters
    :return: pre=trained resNet model
    '''

    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

    last_layer = base_model.output

    new_layer = Dense(units=3, activation='relu')(last_layer)

    predictions = Dense(1, activation='sigmoid')(new_layer)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = SGD(learning_rate=Params['Optimizer']['SGD']['learning_rate'],
                    momentum=Params['Optimizer']['SGD']['momentum'], nesterov=Params['Optimizer']['SGD']['nesterov'])

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model


def train_generator(model, TrainData, Params):
    '''
    train a model with data augmentation
    :param training set (data + labels) and validation set (data+labels)
    :param: Params: pipe parameters
    :return: trained model
    '''

    train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.1, rotation_range=10
                                       , horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train, val = train_test_split(TrainData, test_size=0.3)
    TrainDataRep_x, TrainDataRep_y = prepare(Params, train)
    ValDataRep_x, ValDataRep_y = prepare(Params, val)

    train_generator = train_datagen.flow(TrainDataRep_x, TrainDataRep_y, batch_size=30)
    val_generator = val_datagen.flow(ValDataRep_x, ValDataRep_y, batch_size=30)

    model.fit_generator(train_generator,
                        steps_per_epoch=6,
                        epochs=Params['Model']['epochs'],
                        validation_data=val_generator,
                        validation_steps=2,
                        verbose=1)

    return model


def train_generator_CV(model, TrainDataRep_x, TrainDataRep_y, ValDataRep_x, ValDataRep_y, Params):
    '''
     train a model with data augmentation + special case for cross-validation
     :param training set (data + labels) and validation set (data+labels)
     :param: Params: pipe parameters
     :return: trained model
     '''

    train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.1, rotation_range=10
                                       , horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # train, val = train_test_split(TrainData, test_size=0.5)
    # TrainDataRep_x, TrainDataRep_y = prepare(Params, TrainDataRep)
    # ValDataRep_x, ValDataRep_y = prepare(Params, ValDataRep)

    train_generator = train_datagen.flow(TrainDataRep_x, TrainDataRep_y, batch_size=30)
    val_generator = val_datagen.flow(ValDataRep_x, ValDataRep_y, batch_size=30)

    model.fit_generator(train_generator,
                        steps_per_epoch=5,
                        epochs=Params['Model']['epochs'],
                        validation_data=val_generator,
                        validation_steps=2,
                        verbose=1)

    return model


# </editor-fold>

# <editor-fold desc="resNet + RF model">
def RF_createResNetModel(train_x, train_y, Params):
    '''
    1. creating and model with a single-neuron output + training it
    2. removes the single neuron to have a output of 50 values
    :param train_x: images
    :param train_y: labels
    :param Params: pipe parameters
    :return: trained  model for features extraction
    '''

    print('Building and fitting the ResNet')
    # creating and training a new model with new 100 neurons layer
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

    last_layer = base_model.output

    # add layer with 50 neurons
    new_layer = Dense(50, activation='sigmoid')(last_layer)
    # add the output layer with sigmoid activation
    predictions = Dense(1, activation='sigmoid')(new_layer)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = SGD(learning_rate=Params['Optimizer']['SGD']['learning_rate'],
                    decay=Params['Optimizer']['SGD']['decay'], momentum=Params['Optimizer']['SGD']['momentum'],
                    nesterov=Params['Optimizer']['SGD']['nesterov'])

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    model.fit(train_x, train_y, batch_size=Params['Model']['batch_size'], epochs=Params['Model']['epochs'])

    print('finished training the ResNet')

    # remove the last single-neuron layer to stay with 50 neurons layer
    model.layers.pop()
    model2 = Model(model.input, model.layers[-1].output)

    return model2


def RF_featuresExtraction(resnetModel, TestDataRep_x, TestDataRep_y, addLabel):
    '''
    we use the trained net to represnet each image with a vector of 50 values
   :param resnetModel: trained model
   :param train_x: images
   :param train_y: labels
   :param addLabel: this boolean is used to distinguish between the training phase
                    and the test set (where we just extract features)
   :return: DataFrame, each row is an image and the columns are the 50 values (features)
   '''

    output_values = resnetModel.predict(TestDataRep_x)

    df_columns = []

    # create the column vector
    for i in range(1, 51):
        name = 'F{}'.format(i)
        df_columns.append(name)

    # convert the net output into table base representation
    features_df = pd.DataFrame(output_values, columns=df_columns)

    # only fot training phase
    if addLabel:
        # add the label column
        features_df['label'] = pd.Series(TestDataRep_y, index=features_df.index)

        if False:  # for tuning phase only
            dest_folder = 'C:/Users/idofi/OneDrive/Documents/BGU/masters/year A/computer vision/Task_2'
            pkl_name = 'RF_dataset.pkl'
            features_df.to_pickle('{}/{}'.format(dest_folder, pkl_name))

    return features_df


def RF_train(features_df, Params):
    '''
    train the RandomForest classifier
   :param features_df: dataframe of the images and their features
   :param Params: Pipe parameters
   :return: trained RF classifier
   '''

    print('Start training the RF classifier')
    RF = RandomForestClassifier(random_state=0, max_depth=Params['RF']['max_depth'],
                                n_estimators=Params['RF']['n_estimators'])

    X_set = features_df[features_df.columns[0:50]]
    Y_set = features_df['label']

    RF.fit(X_set, Y_set)
    print('finished training the RF classifier')
    return RF


def RF_test(resnetModel, RFmodel, TestDataRep_x, Params):
    '''
      1. we use the resnet model to extract features out of the images
      2. we predict using the RF classifier
     :param resnetModel: trained net for features extraction
     :param RFmodel: classifier (RandomForest)
     :param TestDataRep_x: test images
     :param Params: Pipe parameters
     :return: predictions (binary + probabilities)
     '''

    print('Creating predictions...')

    # convert test set to RF representation
    x_df = RF_featuresExtraction(resnetModel, TestDataRep_x, None, False)
    # get probabilities
    output_values = RFmodel.predict_proba(x_df)

    # extract only the prob for class 1
    fixed_output_values = []
    for value in output_values:
        fixed_output_values.append(value[1])

    # convert probabilities into binary result
    predictions = convert_to_binary_classification(output_values, Params['Model']['threshold'], 1)

    return np.array(predictions), np.array(fixed_output_values)


# </editor-fold >

# <editor-fold desc="   Hyper-parametres Tuning + Cross validation">


def hyperParametersTuning_baseline(TrainData, Params, learning_rate_r, momentum_r, nesterov_r, epochs_r):
    '''
        Hyper parameters tuning of baseline model
      :param    TrainData: vector of 10 trained classifiers
      :param    Params: Vector with all model parameters
      :param    hyper-parameters lists
      :return   dataframe with results for each configuration of hyper parameters
    '''

    columns = ['epochs', 'Optimizer', 'learning_rate', 'momentum', 'nesterov', 'train_error', 'validation_error']
    main_tune_results = pd.DataFrame(columns=columns)

    # iterate over all possible configurations
    for epochs in epochs_r:
        for learning_rate in learning_rate_r:
            for momentum in momentum_r:
                for nesterov in nesterov_r:
                    new_params = Params.copy()
                    new_params['Model']['epochs'] = epochs
                    new_params['Optimizer']['SGD']['learning_rate'] = learning_rate
                    new_params['Optimizer']['SGD']['momentum'] = momentum
                    new_params['Optimizer']['SGD']['nesterov'] = nesterov

                    print(
                        '\nepochs: {} Optimizer:{}  lr:{} momentum:{} nesterov:{}'.format(epochs, 'SGD', learning_rate,
                                                                                          momentum,
                                                                                          nesterov))
                    result_validation, result_train = KfoldCrossValidation_baseline(new_params, TrainData)
                    print('Validation Error: {}  Train Error: {}'.format(result_validation, result_train))
                    # apply cross-validation and get training and validation error
                    main_tune_results = main_tune_results.append(
                        {'epochs': epochs, 'Optimizer': 'SGD', 'learning_rate': learning_rate, 'momentum': momentum,
                         'nesterov': nesterov, 'train_error': result_train, 'validation_error': result_validation},
                        ignore_index=True)

    return main_tune_results


def hyperParametersTuning_2layer(TrainData, Params, learning_rate_r, momentum_r, nesterov_r):
    '''
    Hyper parameters tuning of 2layer model
    :param    TrainData: vector of 10 trained classifiers
    :param    Params: Vector with all model parameters
    :param    hyper-parameters lists
    :return   dataframe with results for each configuration of hyper parameters
    '''

    columns = ['Optimizer', 'learning_rate', 'momentum', 'nesterov', 'train_error', 'validation_error']
    main_tune_results = pd.DataFrame(columns=columns)

    # iterate over all possible configurations
    for learning_rate in learning_rate_r:
        for momentum in momentum_r:
            for nesterov in nesterov_r:
                new_params = Params.copy()
                print('\nOptimizer:{}  lr:{} momentum:{} nesterov:{}'.format('SGD', learning_rate, momentum, nesterov))
                new_params['Optimizer']['SGD']['learning_rate'] = learning_rate
                new_params['Optimizer']['SGD']['momentum'] = momentum
                new_params['Optimizer']['SGD']['nesterov'] = nesterov

                result_validation, result_train = KfoldCrossValidation_2layer(new_params, TrainData)
                print('Validation Error: {}  Train Error: {}'.format(result_validation, result_train))
                # apply cross-validation and get training and validation error
                main_tune_results = main_tune_results.append(
                    {'Optimizer': 'SGD', 'learning_rate': learning_rate, 'momentum': momentum, 'nesterov': nesterov,
                     'train_error': result_train, 'validation_error': result_validation}, ignore_index=True)

    return main_tune_results


def KfoldCrossValidation_baseline(Params, TrainData):
    '''
     stratified k-fold cross-validation
     :param   TrainData: vector of 10 trained classifiers
     :param   Params: Vector with all model parameters
     :return dataframe with results for each configuration of hyper parameters
    '''

    new_params = Params.copy()

    TrainDataRep_x, TrainDataRep_y = prepare(new_params, TrainData)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    fold = 1
    results_validation = []
    results_train = []

    # apply Cross-Validation using K-fold
    for train_index, test_index in skf.split(TrainDataRep_x, TrainDataRep_y):
        # aug_train_x, aug_train_y = DataAugment(TrainDataRep_x[train_index], TrainDataRep_y[train_index],
        #                                        Params['Prepare']['dataAugment'])

        train_x = TrainDataRep_x[train_index]
        train_y = TrainDataRep_y[train_index]

        model = build_baseline_model(new_params)

        FittedNet = train(model, train_x, train_y, new_params)

        # predict & evaluate on VALIDATION set
        predictions_validation, probs_validation = test(FittedNet, TrainDataRep_x[test_index], new_params)

        Summary_validation = evaluate(predictions_validation, None, TrainDataRep_y[test_index], probs_validation, False)

        # predict & evaluate on TRAIN set
        predictions_train, probs_train = test(FittedNet, train_x, new_params)

        Summary_train = evaluate(predictions_train, None, train_y, probs_train, False)

        # append the  results into a list
        results_validation.append(Summary_validation['error_rate'])
        results_train.append(Summary_train['error_rate'])
        print(' Fold: {} -- validation error: {}  train error: {}'.format(fold, Summary_validation['error_rate'],
                                                                          Summary_train['error_rate']))

        print(Summary_validation['conf_matrix'])

        fold = fold + 1

    final_results_validation = np.mean(results_validation)
    final_results_train = np.mean(results_train)

    return final_results_validation, final_results_train


def KfoldCrossValidation_2layer(Params, TrainData):
    '''
         stratified k-fold cross-validation
         :param   TrainData: vector of 10 trained classifiers
         :param   Params: Vector with all model parameters
         :return dataframe with results for each configuration of hyper parameters
    '''

    new_params = Params.copy()
    TrainDataRep_x, TrainDataRep_y = prepare(new_params, TrainData)

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    fold = 1
    results_validation = []
    results_train = []

    # apply Cross-Validation using K-fold
    for train_index, test_index in skf.split(TrainDataRep_x, TrainDataRep_y):
        model = build_2layer_model(new_params)

        FittedNet = train_generator_CV(model, TrainDataRep_x[train_index], TrainDataRep_y[train_index],
                                       TrainDataRep_x[test_index], TrainDataRep_y[test_index], new_params)

        # predict & evaluate on VALIDATION set
        predictions_validation, probs_validation = test(FittedNet, TrainDataRep_x[test_index], new_params)
        Summary_validation = evaluate(predictions_validation, None, TrainDataRep_y[test_index], probs_validation, False)

        # predict & evaluate on TRAIN set
        predictions_train, probs_train = test(FittedNet, TrainDataRep_x[train_index], new_params)
        Summary_train = evaluate(predictions_train, None, TrainDataRep_y[train_index], probs_train, False)

        # append the  results into a list
        results_validation.append(Summary_validation['error_rate'])
        results_train.append(Summary_train['error_rate'])
        print(' Fold: {} -- validation error: {}  train error: {}'.format(fold, Summary_validation['error_rate'],
                                                                          Summary_train['error_rate']))

        fold = fold + 1

    # calc the avg error for both validation and training
    # print(results_validation)
    # print(results_train)

    final_results_validation = np.mean(results_validation)
    final_results_train = np.mean(results_train)

    return final_results_validation, final_results_train


def tune_Net_RF():
    '''
     use for tuning the RF_resnet Model
    '''
    np.random.seed(0)
    Params = GetDefaultParameters()

    data_path = 'C:/Users/idofi/OneDrive/Documents/BGU\masters/year A/computer vision/Task_2/FlowerData'
    test_images_indices = list(range(301, 473))

    Params['Data']['path'] = data_path
    Params['Data']['test_images_indices'] = test_images_indices

    start_time = time.time()

    # DandL = GetData(Params, False)
    DandL = pd.read_pickle('raw_data.pkl')

    TrainData, TestData = TrainTestSplit(Params, DandL)
    TrainDataRep_x, TrainDataRep_y = prepare(Params, TrainData)

    v_moment = [0.8, 0.9, 0.95]
    v_lr = [0.5]
    for mom in range(0, len(v_moment)):
        for lr in range(0, len(v_lr)):
            lr1 = v_lr[lr]
            mom1 = v_moment[mom]

            Params['Optimizer']['SGD']['learning_rate'] = lr1
            Params['Optimizer']['SGD']['momentum'] = mom1

            X_train, X_val, y_train, y_val = train_test_split(TrainDataRep_x, TrainDataRep_y, test_size=0.3,
                                                              random_state=42)

            resNet_model = RF_createResNetModel(X_train, y_train, Params)

            features_df = RF_featuresExtraction(resNet_model, X_train, y_train, True)

            RF_model = RF_train(features_df, Params)

            predictions_RF, output_values_RF = RF_test(resNet_model, RF_model, X_val, Params)

            Summary = evaluate(predictions_RF, None, y_val, output_values_RF, displayWorst=False)
            ReportResults(Summary, output_values_RF, y_val, False, start_time, Params)


def tuning():
    '''
     use for tuning the baseline model
    '''
    np.random.seed(0)
    Params = GetDefaultParameters()

    data_path = 'C:/Users/idofi/OneDrive/Documents/BGU\masters/year A/computer vision/Task_2/FlowerData'
    test_images_indices = list(range(301, 473))

    Params['Data']['path'] = data_path
    Params['Data']['test_images_indices'] = test_images_indices

    DandL = GetData(Params, False)
    # DandL = pd.read_pickle('fold1_raw_data.pkl')

    TrainData, TestData = TrainTestSplit(Params, DandL)
    print('Train - Test Sizes:\ntrain: {} \ntest: {}'.format(TrainData.shape[0], TestData.shape[0]))

    Params['Model']['batch_size'] = 32
    Params['Model']['epochs'] = 5

    # SGD
    epochs_r = [1, 2, 3, 4, 5, 6, 7]
    epochs_r = [5]
    learning_rate_r = [0.001, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
    learning_rate_r = [0.4]
    momentum_r = [0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9]
    nesterov_r = [False]
    tune_results = hyperParametersTuning_baseline(TrainData, Params, learning_rate_r, momentum_r, nesterov_r, epochs_r)

    sorted_TuningTable = tune_results.sort_values(['validation_error']).reset_index(drop=True)
    print(sorted_TuningTable)

    if False:
        dest_folder = 'C:/Users/idofi/OneDrive/Documents/BGU/masters/year A/computer vision/Task_2'
        pkl_name = 'SGD_momentum_tuning.pkl'
        # tune_results.to_pickle('{}/{}'.format(dest_folder, pkl_name))
        tune_results = pd.read_pickle('{}/{}'.format(dest_folder, pkl_name))

    plt.plot(tune_results['momentum'], tune_results['validation_error'], marker='o', color='b')
    plt.plot(tune_results['momentum'], tune_results['train_error'], marker='o', color='g')
    plt.xlabel('momentum')
    plt.ylabel('Error rate')
    plt.title("SGD momentum tuning", fontsize=12, fontweight=0, color='black')
    plt.legend(['validation', 'training'])
    plt.show()


def tuning_2():
    '''
     use for tuning the 2-layer model Model
    '''
    np.random.seed(0)
    Params = GetDefaultParameters()

    data_path = 'C:/Users/idofi/OneDrive/Documents/BGU\masters/year A/computer vision/Task_2/FlowerData'
    test_images_indices = list(range(301, 473))

    Params['Data']['path'] = data_path
    Params['Data']['test_images_indices'] = test_images_indices

    DandL = GetData(Params, False)
    # DandL = pd.read_pickle('fold1_raw_data.pkl')

    TrainData, TestData = TrainTestSplit(Params, DandL)
    print('Train - Test Sizes:\ntrain: {} \ntest: {}'.format(TrainData.shape[0], TestData.shape[0]))

    Params['Model']['batch_size'] = 32
    Params['Model']['epochs'] = 1

    # SGD
    learning_rate_r = [0.001, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
    learning_rate_r = [0.7]
    momentum_r = [0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9]
    momentum_r = [0.3, 0.4, 0.6]
    nesterov_r = [False]
    tune_results = hyperParametersTuning_2layer(TrainData, Params, learning_rate_r, momentum_r, nesterov_r)

    sorted_TuningTable = tune_results.sort_values(['validation_error']).reset_index(drop=True)
    # print(sorted_TuningTable)

    if False:
        dest_folder = 'C:/Users/idofi/OneDrive/Documents/BGU/masters/year A/computer vision/Task_2'
        pkl_name = '2layer_SGD_momentum_tuning.pkl'
        # tune_results.to_pickle('{}/{}'.format(dest_folder, pkl_name))
        tune_results = pd.read_pickle('{}/{}'.format(dest_folder, pkl_name))

    plt.plot(tune_results['momentum'], tune_results['validation_error'], marker='o', color='b')
    plt.plot(tune_results['momentum'], tune_results['train_error'], marker='o', color='g')
    plt.xlabel('momentum')
    plt.ylabel('Error rate')
    plt.title("SGD SGD_momentum_tuning tuning", fontsize=12, fontweight=0, color='black')
    plt.legend(['validation', 'training'])
    plt.show()


def cross_val_rand(Train_set):
    '''
     cross-validation for the RF model seperatly
      :param   Train_set: training set
    '''

    # Train_set = pd.read_pickle('{}\{}'.format('C:\\Users\\yonat\\Desktop\\Class\\YEAR D\\Semester A\\VR\\Project\\Task_2', 'RF_dataset.pkl'))
    print(Train_set)
    X = Train_set[Train_set.columns[:-1]]
    Y = Train_set['label']
    # DepthSearch(range(1,20),X,Y)
    clf = RandomForestClassifier(max_depth=6, random_state=1438)
    plot_validation_curve(X, Y, clf, range(1, 50), param_name='n_estimators')


def DepthSearch(range, X, Y):
    '''
    Tuning the RF depth + plotting the results
    :param   range: range of tree depth
    :param   Y: labels
    :param   X: images
    '''
    RNFRes2 = pd.DataFrame(columns=['Max-depth', 'Mean_Score_Test', 'Mean_Score_Train'])
    for i in range:
        RNF = RandomForestClassifier(n_estimators=20, max_depth=i, random_state=1438)
        cv_results_RNF = cross_validate(RNF, X, Y, cv=8, return_train_score=True)
        Mean_Score_Train_RNF = np.mean(cv_results_RNF['train_score'])
        Mean_Score_Test_RNF = np.mean(cv_results_RNF['test_score'])
        RNFRes2 = RNFRes2.append(
            {'Max-depth': i, 'Mean_Score_Test': Mean_Score_Test_RNF, 'Mean_Score_Train': Mean_Score_Train_RNF},
            ignore_index=True)

    print(RNFRes2)


def plot_validation_curve(X, y, estimator, param_range, title='Valdation_curve', alpha=0.1,
                          scoring='accuracy', param_name="max_depth", cv=10, save=False, rotate=False):
    '''
    plot the cross validation tunning phase
    '''
    train_scores, test_scores = validation_curve(estimator,
                                                 X, y, param_name=param_name, cv=cv,
                                                 param_range=param_range, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    best_param_value = max(test_scores_mean)
    best_param = param_range[list(test_scores_mean).index(best_param_value)]

    plt.figure(figsize=(15, 15))
    sort_idx = np.argsort(param_range)
    param_range = np.array(param_range)[sort_idx]
    train_mean = np.mean(train_scores, axis=1)[sort_idx]
    train_std = np.std(train_scores, axis=1)[sort_idx]
    test_mean = np.mean(test_scores, axis=1)[sort_idx]
    test_std = np.std(test_scores, axis=1)[sort_idx]
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)

    plt.title(title + "- Best {} is {} with {} CV mean score of {}".format(param_name, best_param, scoring,
                                                                           round(best_param_value, 3)))

    plt.grid(ls='--')
    plt.xlabel(param_name)
    plt.xticks(param_range)
    if rotate:
        plt.xticks(rotation='vertical')
    plt.ylabel('Average values and standard deviation')
    plt.legend(loc='best')

    print(
        "Best {} is {} with {} CV mean score of {}".format(param_name, best_param, scoring, round(best_param_value, 3)))
    if save:
        plt.savefig('Plots/' + title + '_' + param_name)
    plt.show()


# </editor-fold>

# <editor-fold desc="Global functions">


def convert_to_binary_classification(output_values, threshold, valueIndx):
    '''
     converts float values of prediction into binary result
     based on a given threshold
     :param output_values: float values of prediction
     :param threshold: thershold for decision
     :param valueIndx: determine what is the index of the right values
     :return: predition in binary result
   '''

    predictions = []

    for value in output_values:
        if value[valueIndx] > threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions


def test(model, TestDataRep_x, Params):
    '''
      perform prediction of the trained model
      :param Model: trained classifier
      :param TestDataRep_x: testing feature vectors
      :param: Params: pipe parameters
      :return: predition (0/1) + the model output values (float)
    '''

    output_values = model.predict(TestDataRep_x)
    predictions = convert_to_binary_classification(output_values, Params['Model']['threshold'], 0)

    return np.array(predictions), output_values


def evaluate(Results, TestData, TestDataRep_y, output_values, displayWorst):
    '''
         calculate error +confusion matrix and display worst classified images
         :param Results: predictions vector
         :param TestDataRep_y: real values vector
         :param output_values: probabilities vector
         :param displayWorst: boolean
         :return: dictionary with the error rate and the confusion matrix
   '''

    error_rate = round(1 - accuracy_score(TestDataRep_y, Results), 3)
    conf_matrix = confusion_matrix(TestDataRep_y, Results, labels=[1, 0])

    summary = {'error_rate': error_rate, 'conf_matrix': conf_matrix}

    if displayWorst:
        PredsAndTrue = pd.DataFrame(columns=['Pred', 'Real', 'Prob'])
        PredsAndTrue['Pred'] = Results
        PredsAndTrue['Real'] = TestDataRep_y
        PredsAndTrue['Prob'] = output_values
        PredsAndTrue = PredsAndTrue.sort_values(['Prob'])
        indexNames = PredsAndTrue[(PredsAndTrue['Pred']) == (PredsAndTrue['Real'])].index
        PredsAndTrue.drop(indexNames, inplace=True)

        PredsAndTrue0 = PredsAndTrue[PredsAndTrue['Pred'] == 0]
        PredsAndTrue1 = PredsAndTrue[PredsAndTrue['Pred'] == 1]

        PredsAndTrue1 = PredsAndTrue1.sort_values(['Prob'], ascending=False)
        TopPredsAndTrue0 = PredsAndTrue0.head()
        TopPredsAndTrue1 = PredsAndTrue1.head()

        fig = plt.figure(figsize=(12, 12))
        fig.suptitle('Worst 5 Mistakes in each error type', fontsize="x-large")
        loc = 1
        TestData = TestData.reset_index()
        err_idx_1 = 0
        for idx in TopPredsAndTrue0.index:
            err_idx_1 = err_idx_1 + 1
            fig.add_subplot(2, 5, loc)
            loc = loc + 1
            imsize = 224
            # print(TestData['Data'][idx].shape)
            image = cv2.resize(TestData['Data'][idx], (imsize, imsize))
            mig = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            tit = 'error-index: {} \nImage #:{}\nTrue:{} Pred:{} \n Score{}'.format(err_idx_1, TestData['Img_ID'][idx],
                                                                                    TopPredsAndTrue0.iat[loc - 2, 1],
                                                                                    TopPredsAndTrue0.iat[loc - 2, 0],
                                                                                    round(TopPredsAndTrue0.iat[
                                                                                              loc - 2, 2], 2))
            plt.title(tit, fontsize=9)
            mig.axes.get_xaxis().set_visible(False)
            mig.axes.get_yaxis().set_visible(False)

        loc = 6
        err_idx_2 = 0
        for idx in TopPredsAndTrue1.index:
            err_idx_2 = err_idx_2 + 1
            fig.add_subplot(2, 5, loc)
            loc = loc + 1
            imsize = 224
            image = cv2.resize(TestData['Data'][idx], (imsize, imsize))
            mig = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            tit = 'error-index: {} \nImage #:{}\nTrue:{} Pred:{} \n Score{}'.format(err_idx_2, TestData['Img_ID'][idx],
                                                                                    TopPredsAndTrue1.iat[loc - 7, 1],
                                                                                    TopPredsAndTrue1.iat[loc - 7, 0],
                                                                                    round(TopPredsAndTrue1.iat[
                                                                                              loc - 7, 2], 2))
            plt.title(tit, fontsize=9)
            mig.axes.get_xaxis().set_visible(False)
            mig.axes.get_yaxis().set_visible(False)

        plt.show()

    return summary


def ReportResults(Summary, output_values, TestDataRep_y, displayPlots, start_time):
    '''
    printing the summary of results and display plots (conf_matrix + precision-recall curve)
    :param   Summary: dictionary with train error,validation error,Confusion matrix
    :param   output_values: vector of predictions
    :param   TestDataRep_y: test set labels
    :param   displayPlots: boolean
    :param   start_time: starting time of program
    :return None
    '''

    print('\n')
    print('|---------------------------------------|')
    print('|-------------|  Results  |-------------|')
    print('|---------------------------------------|')

    print('\nTest Error: {}'.format(Summary['error_rate']))
    print('\nConfusion Matrix: \n{}'.format(Summary['conf_matrix']))

    # printRunTime(start_time)

    if displayPlots:
        # plot the precision-recall curve
        precision, recall, _ = precision_recall_curve(TestDataRep_y, output_values, pos_label=1)
        # plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        # plt.style.use('dark_background')
        plt.plot(recall, precision, color='C4')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve')

        # plot the confusion matrix
        ax = plt.subplot(2, 2, 2)
        # plt.style.use('dark_background')
        sns.heatmap(Summary['conf_matrix'], annot=True, ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['Flower', 'Not Flower'])
        ax.yaxis.set_ticklabels(['Flower', 'Not Flower'])

        fig.subplots_adjust(wspace=0.5)
        fig.suptitle('Results Report')
        plt.show()


def printRunTime(strat_time):
    '''
     print the total running time
     :param   strat_time: starting timestamp
     '''

    end = time.time()
    temp = end - strat_time
    minutes = temp // 60
    seconds = temp - 60 * minutes
    print('\nTotal Run Time: %d:%d minutes' % (minutes, seconds))


# </editor-fold>


def main_baseline(dataPath, testImagesIndices):
    np.random.seed(0)
    Params = GetDefaultParameters()

    data_path = dataPath
    test_images_indices = testImagesIndices

    Params['Data']['path'] = data_path
    Params['Data']['test_images_indices'] = test_images_indices



    start_time = time.time()
    DandL = GetData(Params, False)

    TrainData, TestData = TrainTestSplit(Params, DandL)

    TrainDataRep_x, TrainDataRep_y = prepare(Params, TrainData)
    TestDataRep_x, TestDataRep_y = prepare(Params, TestData)

    # data augmentation
    TrainDataRep_x, TrainDataRep_y = DataAugment(TrainDataRep_x, TrainDataRep_y, Params['Prepare']['dataAugment'])

    model = build_baseline_model(Params)

    trained_model = train(model, TrainDataRep_x, TrainDataRep_y, Params)

    predictions, output_values = test(trained_model, TestDataRep_x, Params)

    Summary = evaluate(predictions, TestData, TestDataRep_y, output_values, displayWorst=False)

    ReportResults(Summary, output_values, TestDataRep_y, True, start_time)


def main_generator(dataPath, testImagesIndices):
    np.random.seed(0)
    Params = GetDefaultParameters()

    data_path = testImagesIndices
    test_images_indices = dataPath

    Params['Data']['path'] = data_path
    Params['Data']['test_images_indices'] = test_images_indices

    start_time = time.time()
    DandL = GetData(Params, False)
    # DandL = pd.read_pickle('raw_data.pkl')

    TrainData, TestData = TrainTestSplit(Params, DandL)
    print('Train - Test Sizes:\ntrain: {} \ntest: {}'.format(TrainData.shape[0], TestData.shape[0]))

    TestDataRep_x, TestDataRep_y = prepare(Params, TestData)

    model = build_2layer_model(Params)

    trained_model = train_generator(model, TrainData, Params)

    predictions, output_values = test(trained_model, TestDataRep_x, Params)

    Summary = evaluate(predictions, TestData, TestDataRep_y, output_values, True)

    ReportResults(Summary, output_values, TestDataRep_y, True, start_time)


def main_RF(dataPath, testImagesIndices):
    np.random.seed(0)
    Params = GetDefaultParameters()

    data_path = dataPath
    test_images_indices = testImagesIndices

    Params['Data']['path'] = data_path
    Params['Data']['test_images_indices'] = test_images_indices

    start_time = time.time()

    DandL = GetData(Params, False)

    TrainData, TestData = TrainTestSplit(Params, DandL)

    TrainDataRep_x, TrainDataRep_y = prepare(Params, TrainData)
    TestDataRep_x, TestDataRep_y = prepare(Params, TestData)

    # fit the ResNetV2 model
    resNet_model = RF_createResNetModel(TrainDataRep_x, TrainDataRep_y, Params)

    # extract features using the resnet model
    features_df = RF_featuresExtraction(resNet_model, TrainDataRep_x, TrainDataRep_y, True)

    # fit the RF model
    RF_model = RF_train(features_df, Params)

    predictions_RF, output_values_RF = RF_test(resNet_model, RF_model, TestDataRep_x, Params)

    Summary = evaluate(predictions_RF, TestData, TestDataRep_y, output_values_RF, displayWorst=False)
    ReportResults(Summary, output_values_RF, TestDataRep_y, True, start_time)


if __name__ == '__main__':
    # please change the following values
    data_path = 'C:/Users/idofi/OneDrive/Documents/BGU\masters/year A/computer vision/Task_2/FlowerData'
    test_images_indices = list(range(301, 473))

    # Start the Pipe-line
    main_RF(data_path, test_images_indices)
