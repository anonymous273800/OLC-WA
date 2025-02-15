from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
from Models.BatchClassification import BatchClassification
from Utils import Predictions, Measures, Util, PredictionsMultiClass
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
import itertools

def olc_wa_mc_ovo_classification(X, y, w_inc, base_model_size, increment_size):
    n_classes = len(np.unique(y))
    w_base = 1 - w_inc

    n_samples, n_features = X.shape
    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    # Step 1: Compute the base model for each pair of classes
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]

    class_pairs = list(itertools.combinations(range(n_classes), 2))  # Create class pairs
    base_classifiers = {}

    for class_pair in class_pairs:
        class_1, class_2 = class_pair

        # Prepare binary labels for class 1 vs class 2
        mask = np.isin(base_model_training_y, [class_1, class_2])
        binary_y = np.where(base_model_training_y[mask] == class_1, 1, 0)

        # Train a binary classifier for this pair
        base_model_w, base_model_b = BatchClassification.logistic_regression_OLC_WA(base_model_training_X[mask], binary_y)
        base_coeff = np.array(np.append(np.append(base_model_w, -1), base_model_b))
        base_classifiers[class_pair] = base_coeff

    base_classifiers_converted = np.array(list(base_classifiers.values()))
    acc = PredictionsMultiClass.compute_acc_olr_wa_ovo(base_model_training_X, base_model_training_y, base_classifiers_converted, n_classes)
    acc_list = np.append(acc_list, acc)
    epoch_list = np.append(epoch_list, no_of_base_model_points)

    # Step 2: Incremental learning for each mini-batch
    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]

        for class_pair in class_pairs:
            class_1, class_2 = class_pair

            # Prepare binary labels for class 1 vs class 2
            mask = np.isin(yj, [class_1, class_2])
            binary_y = np.where(yj[mask] == class_1, 1, 0)

            # Train incremental model for this pair
            inc_model_w, inc_model_b = BatchClassification.logistic_regression_OLC_WA(Xj[mask], binary_y)
            inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))

            # Fetch base coefficients
            base_coeff = base_classifiers[class_pair]

            # Extract n1, n2, d1, d2 for intersection computation
            n1 = base_coeff[:-1]
            n2 = inc_coeff[:-1]
            d1 = base_coeff[-1]
            d2 = inc_coeff[-1]

            if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

            n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
            n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization

            avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

            intersection_point = PlanesIntersection.find_intersection_hyperplaneND(
                n1=n1, n2=n2, d1=d1, d2=d2, w_base=w_base, w_inc=w_inc)

            avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
            base_classifiers[class_pair] = avg_plane

            # # Update cost and epoch lists (optional)
            # inc_predicted_y_test = Predictions.predict_(Xj[mask], base_classifiers[class_pair])
            # cost = Measures.cross_entopy_cost(binary_y, inc_predicted_y_test)
            # cost_list = np.append(cost_list, cost)
            # epoch_list = np.append(epoch_list, i + no_of_base_model_points)
        base_classifiers_converted = np.array(list(base_classifiers.values()))
        acc = PredictionsMultiClass.compute_acc_olr_wa_ovo(Xj, yj, base_classifiers_converted, n_classes)
        acc_list = np.append(acc_list, acc)
        epoch_list = np.append(epoch_list, i+increment_size)

    base_classifiers = np.array(list(base_classifiers.values()))  # Convert dict to list and then to array
    return base_classifiers, epoch_list, cost_list, acc_list




def olc_wa_mc_ovo_classification_KFold(X, y, w_inc, base_model_size, increment_size, seed):
    # kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        coeff_classifiers, epoch_list, cost_list, acc_list = olc_wa_mc_ovo_classification(X_train, y_train, w_inc, base_model_size, increment_size)

        # predicted_y_test = np.zeros(X_test.shape[0])
        # for i in range(X_test.shape[0]):
        #     # Get the predicted probabilities for each classifier
        #     probabilities = np.array([Predictions.predict_prob_((X_test[i].reshape(1, -1)), clf) for clf in coeff_classifiers])
        #     predicted_y_test[i] = np.argmax(probabilities)
        predicted_y_test = np.zeros(X_test.shape[0])
        n_classes = len(np.unique(y_train))  # Determine the number of classes
        class_pairs = list(itertools.combinations(range(n_classes), 2))  # Generate class pairs

        for i in range(X_test.shape[0]):
            # Initialize a vote counter for each class
            votes = np.zeros(n_classes)

            # For each test sample, get the prediction from each classifier
            for clf_idx, clf in enumerate(coeff_classifiers):
                class_1, class_2 = class_pairs[clf_idx]  # Get the class pair for this classifier
                probability = Predictions.predict_prob_((X_test[i].reshape(1, -1)), clf)

                # For binary classification in OvO, assign the vote to the winning class
                if probability > 0.5:  # If probability > 0.5, vote for class_1
                    votes[class_1] += 1
                else:  # If probability <= 0.5, vote for class_2
                    votes[class_2] += 1

            # Assign the final predicted class based on the highest vote
            predicted_y_test[i] = np.argmax(votes)

        acc = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc)

    return np.array(scores).mean()

##############################################LSTM-SCCM--------------------------------------------------------
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


def add_mini_batch_statistics_to_memory(Xj, yj, base_classifiers, memoryManager, recomputed, n_classes):
    acc = PredictionsMultiClass.compute_acc_olr_wa_ovo(Xj, yj, base_classifiers, n_classes=n_classes)
    if recomputed:
        print("\t recomputed current mini-batch acc ", acc)
    else:
        print("\t current mini-batch initial acc ", acc)

    miniBatchMetaData = MiniBatchMetaData(None, None, acc)
    memoryManager.add_mini_batch_data(miniBatchMetaData)


def surface_level_retrain_using_tuned_hyperparameters(w_inc_tuned, norms1, norms2, intersection_points):
    w_base_tuned = round(1 - w_inc_tuned, 10)
    avg_planes = []
    for n1, n2, inter in zip(norms1, norms2, intersection_points):
        avg = (w_base_tuned * n1 + w_inc_tuned * n2) / (w_base_tuned + w_inc_tuned)
        avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, inter)
        avg_planes.append(avg_plane)

    return avg_planes


def train(Xj, yj, base_classifiers, w_inc, w_base, n_classes):
    inc_classifiers = {}
    class_pairs = list(itertools.combinations(range(n_classes), 2))
    for class_pair in class_pairs:
        class_1, class_2 = class_pair
        mask = np.isin(yj, [class_1, class_2])
        binary_y = np.where(yj[mask] == class_1, 1, 0)
        inc_model_w, inc_model_b = BatchClassification.logistic_regression_OLC_WA(Xj[mask], binary_y)
        inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
        inc_classifiers[class_pair] = inc_coeff

    # inc_classifiers = []
    # for class_label in range(n_classes):
    #     yj_binary = np.where(yj == class_label, 1, 0)  # once equal it will put 1 otherwise 0
    #     inc_model_w, inc_model_b = BatchClassification.logistic_regression_OLC_WA(Xj, yj_binary)
    #     inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
    #     inc_classifiers.append(inc_coeff)

    inc_classifiers = np.array(list(inc_classifiers.values()))
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


def olc_wa_lstm_sccm_mc_ovo_classification(X, y, w_inc, base_model_size, increment_size, kpi, multiplier, expr):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()
    n_classes = len(np.unique(y))
    n_samples, n_features = X.shape
    epoch_list = np.array([])
    acc_list = np.array([])

    # Step 1: Compute the base model for each pair of classes
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]

    class_pairs = list(itertools.combinations(range(n_classes), 2))  # Create class pairs
    base_classifiers = {}
    for class_pair in class_pairs:
        class_1, class_2 = class_pair
        # Prepare binary labels for class 1 vs class 2
        mask = np.isin(base_model_training_y, [class_1, class_2])
        binary_y = np.where(base_model_training_y[mask] == class_1, 1, 0)
        # Train a binary classifier for this pair
        base_model_w, base_model_b = BatchClassification.logistic_regression_OLC_WA(base_model_training_X[mask],
                                                                                    binary_y)
        base_coeff = np.array(np.append(np.append(base_model_w, -1), base_model_b))
        base_classifiers[class_pair] = base_coeff

    base_classifiers_converted = np.array(list(base_classifiers.values()))
    base_acc = PredictionsMultiClass.compute_acc_olr_wa_ovo(base_model_training_X, base_model_training_y,base_classifiers_converted, n_classes)
    acc_list = np.append(acc_list, base_acc)
    epoch_list = np.append(epoch_list, no_of_base_model_points)
    miniBatchMetaData = MiniBatchMetaData(None, None, base_acc)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    # Step 2: for t ← 1 to T do
    num_intervals = 5
    mini_batch_generator = get_next_mini_batch(X, y, no_of_base_model_points, increment_size)
    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:

        epoch_list = np.append(epoch_list, (iteration * increment_size) + no_of_base_model_points)
        w_inc = .5
        w_base = .5
        norms1 = []
        norms2 = []
        intersection_points = []
        for class_pair in class_pairs:
            class_1, class_2 = class_pair
            # Prepare binary labels for class 1 vs class 2
            mask = np.isin(yj, [class_1, class_2])
            binary_y = np.where(yj[mask] == class_1, 1, 0)
            # Train incremental model for this pair
            inc_model_w, inc_model_b = BatchClassification.logistic_regression_OLC_WA(Xj[mask], binary_y)
            inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))

            # Fetch base coefficients
            base_coeff = base_classifiers[class_pair]

            # Extract n1, n2, d1, d2 for intersection computation
            n1 = base_coeff[:-1]
            n2 = inc_coeff[:-1]
            d1 = base_coeff[-1]
            d2 = inc_coeff[-1]

            if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue
            n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
            n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization

            avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
            intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2, w_base=w_base, w_inc=w_inc)
            avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
            base_classifiers[class_pair] = avg_plane
            norms1.append(n1norm)
            norms2.append(n2norm)
            intersection_points.append(intersection_point)


        base_classifiers_converted = np.array(list(base_classifiers.values()))
        add_mini_batch_statistics_to_memory(Xj, yj, base_classifiers_converted, memoryManager, recomputed=False,n_classes=n_classes)
        if (len(memoryManager.mini_batch_data) >= 4):
            print("********** SHORT TERM ***********")
            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(memoryManager.mini_batch_data,
                                                                   kpi)  # contains last 4 elements
            print("KPI_Window_ST", KPI_Window_ST)
            threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = conceptDriftDetector.get_meaures(
                KPI_Window_ST, multiplier, kpi)
            print("threshold", threshold, "mean", mean_kpi, "prev", KPI_Window_ST[-2], "curr", KPI_Window_ST[-1],
                  "lower_limit_deviated_kpi", lower_limit_deviated_kpi, "drift_magnitude", drift_magnitude)
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
                for idx, key in enumerate(base_classifiers.keys()):
                    base_classifiers[key] = avg_planes[idx]
                avg_planes_arr = np.array(avg_planes)
                add_mini_batch_statistics_to_memory(Xj, yj, avg_planes_arr, memoryManager, recomputed=True,n_classes=n_classes)

                # 4. Check for LT Drift
                KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(memoryManager.mini_batch_data,kpi)  # whole list 11 entries
                threshold_lt, mean_kpi_lt, std_kpi, lower_limit_deviated_kpi_lt, drift_magnitude_lt = conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
                LT_drift_detected = conceptDriftDetector.detect_LT_drift(KPI_Window_LT, mean_kpi_lt, threshold_lt,kpi)
                print("Long Term Drift Detected", LT_drift_detected)
                if LT_drift_detected:
                    print("INSIDE LONG TERM")
                    print('tuned_w_inc: ', tuned_w_inc)
                    tuned_w_base = round(1 - tuned_w_inc, 10)
                    counter = 0
                    max_no_of_mini_batches_requests = 5
                    while (LT_drift_detected and counter < max_no_of_mini_batches_requests):
                        counter += 1
                        print("\t inside while: additional mini-batch request #", counter)
                        # 1. remove last element in the mini_batch_data
                        memoryManager.remove_last_mini_batch_data()
                        memoryManager.model_is_same_at_this_point()
                        # 2. call next mini-batch, use it to retrain.
                        try:
                            iteration, batch_id, next_Xj, next_yj = next(mini_batch_generator)
                            print("\t additional mini-batch # ", iteration)
                            Xj = next_Xj
                            yj = next_yj
                            avg_planes = train(Xj, yj, avg_planes, tuned_w_inc, tuned_w_base, n_classes)
                            for idx, key in enumerate(base_classifiers.keys()):
                                base_classifiers[key] = avg_planes[idx]
                            #avg_planesX = avg_planes
                            # base_classifiers = avg_planes
                            avg_planes_arr = np.array(avg_planes)
                            add_mini_batch_statistics_to_memory(Xj, yj, avg_planes_arr, memoryManager, recomputed=True, n_classes=n_classes)
                            KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(memoryManager.mini_batch_data,
                                                                                   kpi)  # whole list 11 entries
                            threshold_lt, mean_kpi_lt, std_kpi, lower_limit_deviated_kpi_lt, drift_magnitude_lt = conceptDriftDetector.get_meaures(
                                KPI_Window_LT, multiplier, kpi)
                            LT_drift_detected = conceptDriftDetector.detect_LT_drift(KPI_Window_LT, mean_kpi_lt,
                                                                                     threshold_lt, kpi)
                            epoch_list = np.append(epoch_list, (iteration * increment_size) + no_of_base_model_points)
                            print("\t long_term_drift captured again", LT_drift_detected)
                        except StopIteration:
                            print("End of mini-batch generator reached.")
                            break  # should break the while loop.
                else:
                    print("long term drift not detected")
            else:
                print("short term NOT detected")
            # END CONCEPT DRIFT STUFF
        print("...updating the model...")
        print("=====================================================================================")
        print()
        print()
        print()
        # the model is already updated

        # for idx, key in enumerate(base_classifiers.keys()):
        #     base_classifiers[key] = avg_planesX[idx]



    accuracy_list = memoryManager.get_acc_list()
    cost_list = np.array([])
    base_classifiers_converted = np.array(list(base_classifiers.values()))
    return base_classifiers_converted, epoch_list, cost_list, accuracy_list

