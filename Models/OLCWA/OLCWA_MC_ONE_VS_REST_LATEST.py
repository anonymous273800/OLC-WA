import numpy as np
from Models.BatchClassification import BatchClassification
from Utils import Predictions, Measures, Util
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from Utils import PredictionsMultiClass
from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData

def olc_wa_mc_classification(X, y, w_inc, base_model_size, increment_size):
    n_classes = len(np.unique(y))
    w_base = 1 - w_inc

    n_samples, n_features = X.shape
    cost_list = np.array([])
    epoch_list = np.array([])
    accuracy_list = np.array([])

    # Step 1: Compute the base model, get its coefficients
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    base_classifiers= []
    for class_label in range(n_classes):
        base_model_training_y_binary = np.where(base_model_training_y == class_label, 1, 0)  # once equal it will put 1 otherwise 0
        base_model_w, base_model_b = BatchClassification.logistic_regression_OLC_WA(base_model_training_X,
                                                                             base_model_training_y_binary)
        base_model_predicted_y = Predictions.predict(base_model_training_X, base_model_w, base_model_b)
        base_coeff = np.array(np.append(np.append(base_model_w, -1), base_model_b))
        base_classifiers.append(base_coeff)
        # cost = Measures.cross_entopy_cost(base_model_training_y, base_model_predicted_y)
        # cost_list = np.append(cost_list, cost)
        # epoch_list = np.append(epoch_list, no_of_base_model_points)
        # acc = Measures.accuracy(base_model_training_y, base_model_predicted_y)
        # accuracy_list = np.append(accuracy_list, acc)

    # add epochs, cost, acc to lists (ovr)
    acc = PredictionsMultiClass.compute_acc_olr_wa_ovr(base_model_training_X, base_model_training_y, base_classifiers)
    accuracy_list = np.append(accuracy_list, acc)
    epoch_list = np.append(epoch_list, no_of_base_model_points)




    # Step 2: for t ← 1 to T do
    # In this step we look over the rest of the data incrementally with a determined
    # increment size. In this experiment we use increment_size = max(3, (n+1) * 5) where n is the number of features.
    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        # Step 3: inc-classification = logistic_regression(inc-X,in-y)
        # Calculate the logistic regression for each increment model
        # (for the no of points on each increment increment_size)
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]

        inc_classifiers = []
        for class_label in range(n_classes):
            yj_binary = np.where(yj == class_label, 1, 0)  # once equal it will put 1 otherwise 0
            inc_model_w, inc_model_b = BatchClassification.logistic_regression_OLC_WA(Xj, yj_binary)
            inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
            inc_classifiers.append(inc_coeff)

        # Match each base classifier with the corresponding incremental classifier
        for idx in range(n_classes):
            base_coeff = base_classifiers[idx]
            inc_coeff = inc_classifiers[idx]
            # Extract n1, n2, d1, d2 for intersection computation
            n1 = base_coeff[:-1]  # coefficients of the base model
            n2 = inc_coeff[:-1]  # coefficients of the incremental model
            d1 = base_coeff[-1]  # intercept of the base model
            d2 = inc_coeff[-1]  # intercept of the incremental model

            # in case the base and the incremental models are coincident
            if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

            n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
            n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization

            avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

            # Step 5: intersection-point = get-intersection-point(base-boundary-line, inc-boundary-line)
            # We will find an intersection point between the two models, the base and the incremental.
            # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
            # will be a weighted middle point.
            intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                                   w_base=w_base, w_inc=w_inc)

            # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
            # In this step we define a new space as a result from the average vector 1 and the intersection point
            avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
            base_coeff = avg_plane
            base_classifiers[idx] = base_coeff

        # compute acc (ovr)
        acc = PredictionsMultiClass.compute_acc_olr_wa_ovr(Xj, yj, base_classifiers)
        accuracy_list = np.append(accuracy_list, acc)
        epoch_list = np.append(epoch_list, i + increment_size)

    return base_classifiers, epoch_list, cost_list, accuracy_list




def olc_wa_mc_classification_KFold(X, y, w_inc, base_model_size, increment_size, seed):
    # kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        coeff_classifiers, epoch_list, cost_list, acc_list = olc_wa_mc_classification(X_train, y_train, w_inc, base_model_size, increment_size)

        predicted_y_test = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            # Get the predicted probabilities for each classifier
            probabilities = np.array([Predictions.predict_prob_((X_test[i].reshape(1, -1)), clf) for clf in coeff_classifiers])
            predicted_y_test[i] = np.argmax(probabilities)


        acc = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc)

    return np.array(scores).mean()



######################################LSTM_SCCM#####################################################

def olc_wa_sccm_mc_classification_KFold(X, y, w_inc, base_model_size, increment_size, seed, kpi, multiplier, expr):
    # kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        coeff_classifiers, epoch_list, cost_list, acc_list = olc_wa_sccm_ovr_classification(X_train, y_train, w_inc, base_model_size, increment_size, kpi, multiplier, expr)

        predicted_y_test = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            # Get the predicted probabilities for each classifier
            probabilities = np.array([Predictions.predict_prob_((X_test[i].reshape(1, -1)), clf) for clf in coeff_classifiers])
            predicted_y_test[i] = np.argmax(probabilities)


        acc = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc)

    return np.array(scores).mean()


import uuid
def get_next_mini_batch(X, y, no_of_base_model_points, increment_size):
    n_samples, n_features = X.shape
    j = 0
    # for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
    for i in range(no_of_base_model_points, n_samples, increment_size):
        j = j + 1
        print("*********** mini-batch- ", j, " *************")
        mini_batch_id = uuid.uuid4()  # Generate a new UUID for each mini-batch
        # iteration_number = i // increment_size
        yield j, mini_batch_id, X[i:i + increment_size], y[i:i + increment_size]


def add_mini_batch_statistics_to_memory(Xj, yj, base_classifiers, memoryManager, recomputed, inc_acc):
    acc = PredictionsMultiClass.compute_acc_olr_wa_ovr(Xj, yj, base_classifiers)
    if recomputed:
        print("\t recomputed current mini-batch acc ", acc)
    else:
        print("\t current mini-batch initial acc ", acc)

    inc_acc_updated = False
    if (inc_acc is not None and acc < inc_acc):
        acc = inc_acc
        inc_acc_updated = True
        print("counted one is: ", acc)

    miniBatchMetaData = MiniBatchMetaData(None, None, acc)
    memoryManager.add_mini_batch_data(miniBatchMetaData)
    return acc, inc_acc_updated



# def surface_level_retrain_using_tuned_hyperparameters(w_inc_tuned, n1norm, n2norm, intersection_point):
#
#     w_base_tuned = round(1 - w_inc_tuned, 10)
#
#     avg = (np.dot(w_base_tuned, n1norm) + np.dot(w_inc_tuned, n2norm)) / (w_base_tuned + w_inc_tuned)
#     avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
#
#     return avg_plane

def surface_level_retrain_using_tuned_hyperparameters(w_inc_tuned, norms1, norms2, intersection_points):
    w_base_tuned = round(1 - w_inc_tuned, 10)
    avg_planes = []
    for n1, n2, inter in zip(norms1, norms2, intersection_points):
        avg = (w_base_tuned * n1 + w_inc_tuned * n2) / (w_base_tuned + w_inc_tuned)
        avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, inter)
        avg_planes.append(avg_plane)

    return avg_planes


def train(Xj, yj, base_classifiers, w_inc, w_base, n_classes):
    inc_classifiers = []
    for class_label in range(n_classes):
        yj_binary = np.where(yj == class_label, 1, 0)  # once equal it will put 1 otherwise 0
        inc_model_w, inc_model_b = BatchClassification.logistic_regression_OLC_WA(Xj, yj_binary)
        inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
        inc_classifiers.append(inc_coeff)


    avg_planes = []
    for idx in range(n_classes):
        base_coeff = base_classifiers[idx]
        inc_coeff = inc_classifiers[idx]
        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        # if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization
        # n1norm = n1
        # n2norm = n2

        avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                           w_base=w_base, w_inc=w_inc)
        avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
        avg_planes.append(avg_plane)

    return avg_planes


def olc_wa_lstm_sccm_ovr_classification(X, y, w_inc, base_model_size, increment_size, kpi, multiplier, expr):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()
    n_classes = len(np.unique(y))
    n_samples, n_features = X.shape
    epoch_list = np.array([])
    accuracy_list = np.array([])

    # Step 1: Compute the base model, get its coefficients
    no_of_base_model_points = increment_size #Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    base_classifiers = []
    for class_label in range(n_classes):
        base_model_training_y_binary = np.where(base_model_training_y == class_label, 1, 0)  # once equal it will put 1 otherwise 0
        base_model_w, base_model_b = BatchClassification.logistic_regression_OLC_WA(base_model_training_X, base_model_training_y_binary)
        base_coeff = np.array(np.append(np.append(base_model_w, -1), base_model_b))
        base_classifiers.append(base_coeff)

    # add epochs, cost, acc to lists (ovr)
    base_acc = PredictionsMultiClass.compute_acc_olr_wa_ovr(base_model_training_X, base_model_training_y,
                                                            base_classifiers)
    accuracy_list = np.append(accuracy_list, base_acc)
    epoch_list = np.append(epoch_list, no_of_base_model_points)
    miniBatchMetaData = MiniBatchMetaData(None, None, base_acc)
    memoryManager.add_mini_batch_data(miniBatchMetaData)
    # end Step 1

    # Step 2: for t ← 1 to T do
    num_intervals = 5
    mini_batch_generator = get_next_mini_batch(X, y, no_of_base_model_points, increment_size)
    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:

        epoch_list = np.append(epoch_list, (iteration * increment_size) + no_of_base_model_points)
        # w_inc = .5
        w_base = 1 - w_inc
        inc_classifiers = []
        for class_label in range(n_classes):
            yj_binary = np.where(yj == class_label, 1, 0)  # once equal it will put 1 otherwise 0
            inc_model_w, inc_model_b = BatchClassification.logistic_regression_OLC_WA(Xj, yj_binary)
            inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
            inc_classifiers.append(inc_coeff)

        inc_acc = PredictionsMultiClass.compute_acc_olr_wa_ovr(Xj, yj, inc_classifiers)
        print("INC ACCURACY", inc_acc)
        # Match each base classifier with the corresponding incremental classifier

        norms1 = []
        norms2 = []
        intersection_points = []
        for idx in range(n_classes):
            base_coeff = base_classifiers[idx]
            inc_coeff = inc_classifiers[idx]
            n1 = base_coeff[:-1]  # coefficients of the base model
            n2 = inc_coeff[:-1]  # coefficients of the incremental model
            d1 = base_coeff[-1]  # intercept of the base model
            d2 = inc_coeff[-1]  # intercept of the incremental model

            if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

            n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
            n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization
            avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
            intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2, w_base=w_base, w_inc=w_inc)
            avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
            base_classifiers[idx] = avg_plane
            norms1.append(n1norm)
            norms2.append(n2norm)
            intersection_points.append(intersection_point)


        # Now we have the avg base classifiers, we need to test for acc and LSTM-SCCM
        curr_acc, inc_acc_updated = add_mini_batch_statistics_to_memory(Xj, yj, base_classifiers, memoryManager, recomputed=False, inc_acc=inc_acc)
        if(inc_acc_updated):
            base_classifiers = inc_classifiers

        if (len(memoryManager.mini_batch_data) >= 4):
            print("********** SHORT TERM ***********")
            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(memoryManager.mini_batch_data, kpi)  # contains last 4 elements
            print("KPI_Window_ST", KPI_Window_ST)
            threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = conceptDriftDetector.get_meaures(KPI_Window_ST, multiplier, kpi)
            print("threshold", threshold, "mean", mean_kpi, "prev", KPI_Window_ST[-2], "curr", KPI_Window_ST[-1], "lower_limit_deviated_kpi", lower_limit_deviated_kpi, "drift_magnitude", drift_magnitude)
            ST_drift_detected = conceptDriftDetector.detect_ST_drift(KPI_Window_ST, mean_kpi, threshold, kpi)
            print("SHORT TERM DRIFT DETECTED", ST_drift_detected)
            if (ST_drift_detected):
                print('Short Term Drift Detected')
                # 1. remove last element in the mini_batch_data
                memoryManager.remove_last_mini_batch_data()
                scale = conceptDriftDetector.get_scale(lower_limit_deviated_kpi, mean_kpi, num_intervals, kpi)
                print("scale", scale)
                map_ranges_values = conceptDriftDetector.get_scales_map(scale, expr)
                tuned_w_inc = conceptDriftDetector.get_value_for_range(drift_magnitude, map_ranges_values)
                print('tuned_w_inc', tuned_w_inc)

                # 3. Conduct ST Surface Level Training
                avg_planes = surface_level_retrain_using_tuned_hyperparameters(tuned_w_inc, norms1, norms2, intersection_points)
                curr_acc, new_acc_updated = add_mini_batch_statistics_to_memory(Xj, yj, avg_planes, memoryManager, recomputed=True, inc_acc=curr_acc)
                if new_acc_updated:
                    base_classifiers = avg_planes
            else:
                print("short term NOT detected")
        # END CONCEPT DRIFT STUFF
        print("...updating the model...")
        print("=====================================================================================")
        print()
        print()
        print()
        # the model is already updated


    accuracy_list = memoryManager.get_acc_list()
    cost_list = np.array([])
    return base_classifiers, epoch_list, cost_list, accuracy_list

# only SCCM:

def olc_wa_sccm_ovr_classification(X, y, w_inc, base_model_size, increment_size, kpi, multiplier, expr):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()
    n_classes = len(np.unique(y))
    n_samples, n_features = X.shape
    epoch_list = np.array([])
    accuracy_list = np.array([])

    # Step 1: Compute the base model, get its coefficients
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    base_classifiers = []
    for class_label in range(n_classes):
        base_model_training_y_binary = np.where(base_model_training_y == class_label, 1, 0)  # once equal it will put 1 otherwise 0
        base_model_w, base_model_b = BatchClassification.logistic_regression_OLC_WA(base_model_training_X, base_model_training_y_binary)
        base_coeff = np.array(np.append(np.append(base_model_w, -1), base_model_b))
        base_classifiers.append(base_coeff)

    # add epochs, cost, acc to lists (ovr)
    base_acc = PredictionsMultiClass.compute_acc_olr_wa_ovr(base_model_training_X, base_model_training_y,
                                                            base_classifiers)
    accuracy_list = np.append(accuracy_list, base_acc)
    epoch_list = np.append(epoch_list, no_of_base_model_points)
    miniBatchMetaData = MiniBatchMetaData(None, None, base_acc)
    memoryManager.add_mini_batch_data(miniBatchMetaData)
    # end Step 1

    # Step 2: for t ← 1 to T do
    num_intervals = 5
    mini_batch_generator = get_next_mini_batch(X, y, no_of_base_model_points, increment_size)
    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:

        epoch_list = np.append(epoch_list, (iteration * increment_size) + no_of_base_model_points)
        # w_inc = .1
        w_base = 1 - w_inc
        inc_classifiers = []
        for class_label in range(n_classes):
            yj_binary = np.where(yj == class_label, 1, 0)  # once equal it will put 1 otherwise 0
            inc_model_w, inc_model_b = BatchClassification.logistic_regression_OLC_WA(Xj, yj_binary)
            inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
            inc_classifiers.append(inc_coeff)

        # Match each base classifier with the corresponding incremental classifier

        norms1 = []
        norms2 = []
        intersection_points = []
        for idx in range(n_classes):
            base_coeff = base_classifiers[idx]
            inc_coeff = inc_classifiers[idx]
            n1 = base_coeff[:-1]  # coefficients of the base model
            n2 = inc_coeff[:-1]  # coefficients of the incremental model
            d1 = base_coeff[-1]  # intercept of the base model
            d2 = inc_coeff[-1]  # intercept of the incremental model

            if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

            n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
            n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization
            avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
            intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2, w_base=w_base, w_inc=w_inc)
            avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
            base_classifiers[idx] = avg_plane
            norms1.append(n1norm)
            norms2.append(n2norm)
            intersection_points.append(intersection_point)


        # Now we have the avg base classifiers, we need to test for acc and LSTM-SCCM
        add_mini_batch_statistics_to_memory(Xj, yj, base_classifiers, memoryManager, recomputed=False)

        if (len(memoryManager.mini_batch_data) >= 4):
            print("********** SHORT TERM ***********")
            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(memoryManager.mini_batch_data, kpi)  # contains last 4 elements
            print("KPI_Window_ST", KPI_Window_ST)
            threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = conceptDriftDetector.get_meaures(KPI_Window_ST, multiplier, kpi)
            print("threshold", threshold, "mean", mean_kpi, "prev", KPI_Window_ST[-2], "curr", KPI_Window_ST[-1], "lower_limit_deviated_kpi", lower_limit_deviated_kpi, "drift_magnitude", drift_magnitude)
            ST_drift_detected = conceptDriftDetector.detect_ST_drift(KPI_Window_ST, mean_kpi, threshold, kpi)
            print("SHORT TERM DRIFT DETECTED", ST_drift_detected)
            if (ST_drift_detected):
                print('Short Term Drift Detected')
                # 1. remove last element in the mini_batch_data
                memoryManager.remove_last_mini_batch_data()
                scale = conceptDriftDetector.get_scale(lower_limit_deviated_kpi, mean_kpi, num_intervals, kpi)
                print("scale", scale)
                map_ranges_values = conceptDriftDetector.get_scales_map(scale, expr)
                tuned_w_inc = conceptDriftDetector.get_value_for_range(drift_magnitude, map_ranges_values)
                print('tuned_w_inc', tuned_w_inc)

                # 3. Conduct ST Surface Level Training
                avg_planes = surface_level_retrain_using_tuned_hyperparameters(tuned_w_inc, norms1, norms2, intersection_points)
                add_mini_batch_statistics_to_memory(Xj, yj, avg_planes, memoryManager, recomputed=True)
                base_classifiers = avg_planes
            else:
                print("short term NOT detected")
        # END CONCEPT DRIFT STUFF
        print("...updating the model...")
        print("=====================================================================================")
        print()
        print()
        print()
        # the model is already updated


    accuracy_list = memoryManager.get_acc_list()
    cost_list = np.array([])
    return base_classifiers, epoch_list, cost_list, accuracy_list


