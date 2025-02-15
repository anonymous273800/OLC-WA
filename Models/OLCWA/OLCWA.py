import numpy as np
from Models.BatchClassification import BatchClassification
from Utils import Predictions, Measures, Util
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from sklearn.model_selection import KFold
from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from Utils import Plotter

def olc_wa_classification(X, y, w_inc, base_model_size, increment_size):

    w_base = 1 - w_inc

    n_samples, n_features = X.shape
    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    # Step 1: Compute the base model, get its coefficients
    no_of_base_model_points = increment_size# Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    base_model_w, base_model_b = BatchClassification.logistic_regression(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions.predict(base_model_training_X, base_model_w, base_model_b)
    base_coeff = np.array(np.append(np.append(base_model_w, -1), base_model_b))
    # base_coeff = np.array(np.append(base_model_w, base_model_b))
    cost = Measures.cross_entopy_cost(base_model_training_y, base_model_predicted_y)
    cost_list = np.append(cost_list, cost)
    epoch_list = np.append(epoch_list, no_of_base_model_points)
    base_acc = Measures.accuracy(base_model_training_y, base_model_predicted_y)
    acc_list = np.append(acc_list, base_acc)

    # Step 2: for t ← 1 to T do
    # In this step we look over the rest of the data incrementally with a determined
    # increment size. In this experiment we use increment_size = max(3, (n+1) * 5) where n is the number of features.
    # for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
    for i in range(no_of_base_model_points, n_samples, increment_size):
        # Step 3: inc-classification = logistic_regression(inc-X,in-y)
        # Calculate the logistic regression for each increment model
        # (for the no of points on each increment increment_size)
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]

        inc_model_w, inc_model_b = BatchClassification.logistic_regression(Xj, yj)
        inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
        # inc_coeff = np.array(np.append(inc_model_w, inc_model_b))

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization

        avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        # avg = (np.dot(w_base, n1) + np.dot(w_inc, n2)) / (w_base + w_inc)

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

        inc_predicted_y_test = Predictions.predict_(Xj, base_coeff)
        cost = Measures.cross_entopy_cost(yj, inc_predicted_y_test)
        cost_list = np.append(cost_list, cost)
        epoch_list = np.append(epoch_list, i + no_of_base_model_points)
        inc_acc = Measures.accuracy(yj, inc_predicted_y_test)
        acc_list = np.append(acc_list, inc_acc)

    return base_coeff, epoch_list, cost_list, acc_list

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

def add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed):
    inc_predicted_y_test = Predictions.predict_(Xj, avg_plane)
    acc = Measures.accuracy(yj, inc_predicted_y_test)
    cost = Measures.cross_entopy_cost(yj, inc_predicted_y_test)
    if recomputed:
        print("\t recomputed current mini-batch acc ", acc, "cost", cost)
    else:
        print("\t current mini-batch initial acc ", acc, "cost", cost)

    miniBatchMetaData = MiniBatchMetaData(None, cost, acc)
    memoryManager.add_mini_batch_data(miniBatchMetaData)


def surface_level_retrain_using_tuned_hyperparameters(w_inc_tuned, n1norm, n2norm, intersection_point):
    # retrain
    # w_base_tuned = 1 - w_inc_tuned
    w_base_tuned = round(1 - w_inc_tuned, 10)

    avg = (np.dot(w_base_tuned, n1norm) + np.dot(w_inc_tuned, n2norm)) / (w_base_tuned + w_inc_tuned)
    avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)

    return avg_plane


def train(Xj, yj, base_coeff, w_inc, w_base):
    # r_w_inc = BatchClassification.logistic_regression(Xj, yj)
    inc_model_w, inc_model_b = BatchClassification.logistic_regression(Xj, yj)
    inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
    # inc_coeff = np.array(np.append(inc_model_w, inc_model_b))

    n1 = base_coeff[:-1]
    n2 = inc_coeff[:-1]
    d1 = base_coeff[-1]
    d2 = inc_coeff[-1]

    # in case the base and the incremental models are coincident
    # if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

    # n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
    # n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization
    n1norm = n1
    n2norm = n2

    avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
    intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                           w_base=w_base, w_inc=w_inc)
    avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)

    return avg_plane

def olc_wa_lstm_sccm_classification(X, y, w_inc, base_model_size, increment_size, kpi, multiplier, expr):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    n_samples, n_features = X.shape
    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    # Step 1: Compute the base model, get its coefficients
    no_of_base_model_points = increment_size#Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    base_model_w, base_model_b = BatchClassification.logistic_regression(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions.predict(base_model_training_X, base_model_w, base_model_b)
    base_coeff = np.array(np.append(np.append(base_model_w, -1), base_model_b))
    # Plotter.plot(base_model_training_X, base_model_training_y, base_coeff)
    # base_coeff = np.array(np.append(base_model_w, base_model_b))
    cost = Measures.cross_entopy_cost(base_model_training_y, base_model_predicted_y)
    cost_list = np.append(cost_list, cost)
    epoch_list = np.append(epoch_list, no_of_base_model_points)
    from sklearn.metrics import accuracy_score
    base_acc = Measures.accuracy(base_model_training_y, base_model_predicted_y)
    acc_list = np.append(acc_list, base_acc)

    #############################
    # Fit logistic regression on the subset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    model = LogisticRegression()
    model.fit(base_model_training_X, base_model_training_y)

    # Predict on the training data (since we're using only 20 points)
    y_pred = model.predict(base_model_training_X)

    # Calculate accuracy
    accuracy = accuracy_score(base_model_training_y, y_pred)
    print(f"Accuracy on the first 20 data points: {accuracy:.2f}")

    ############################

    # Statistical Meta-Data Saved about First Mini-Batch (BaseModel)
    miniBatchMetaData = MiniBatchMetaData(None, cost, base_acc)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    num_intervals = 5  # used for the number of intervals in the scaled map.

    # INCREMENTAL_MODEL
    mini_batch_generator = get_next_mini_batch(X, y, no_of_base_model_points, increment_size)
    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:
        # reset to default
        # w_inc = .5
        # w_base = .5
        w_base = 1 - w_inc

        # 1. train using incoming mini-batch
        inc_model_w, inc_model_b = BatchClassification.logistic_regression(Xj, yj)
        inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
        # inc_coeff = np.array(np.append(inc_model_w, inc_model_b))

        model = LogisticRegression()
        model.fit(Xj, yj)

        # Predict on the training data (since we're using only 20 points)
        y_predXX = model.predict(Xj)

        # Calculate accuracy
        accuracy = accuracy_score(yj, y_predXX)
        print(f"Accuracy on the first 20 data points: {accuracy:.2f}")

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization

        avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        # avg = (np.dot(w_base, n1) + np.dot(w_inc, n2)) / (w_base + w_inc)



        # Step 5: intersection-point = get-intersection-point(base-boundary-line, inc-boundary-line)
        # We will find an intersection point between the two models, the base and the incremental.
        # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
        # will be a weighted middle point.
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                               w_base=w_base, w_inc=w_inc)

        # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
        # In this step we define a new space as a result from the average vector 1 and the intersection point
        avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
        # Plotter.plot(Xj, yj, avg_plane)

        inc_predicted_y_test = Predictions.predict_(Xj, avg_plane)
        current_accuracy = Measures.accuracy(yj, inc_predicted_y_test)
        print("MOHA current_accuracy",  current_accuracy)

        # CONCEPT DRIFT STUFF
        # 1. Add statistics about Mini-Batch to memory_manager
        add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=False)

        # 3. The list length min is 3
        if (len(memoryManager.mini_batch_data) >= 4):
            print("********** SHORT TERM ***********")
            # 3. Check for ST Drift
            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(memoryManager.mini_batch_data,
                                                                   kpi)  # contains last 4 elements
            print("KPI_Window_ST", KPI_Window_ST)
            threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = conceptDriftDetector.get_meaures(
                KPI_Window_ST, multiplier, kpi)

            print("threshold", threshold, "mean", mean_kpi, "prev", KPI_Window_ST[-2], "curr", KPI_Window_ST[-1],
                  "lower_limit_deviated_kpi", lower_limit_deviated_kpi, "drift_magnitude", drift_magnitude)
            ST_drift_detected = conceptDriftDetector.detect_ST_drift(KPI_Window_ST, mean_kpi, threshold, kpi)
            print("SHORT TERM DRIFT DETECTED", ST_drift_detected)
            # ST_drift_detected = conceptDriftDetector.detect(memoryManager.mini_batch_data, recomputed=False)
            if (ST_drift_detected):
                print('Short Term Drift Detected')
                # 1. remove last element in the mini_batch_data
                # memoryManager.remove_last_mini_batch_data()

                scale = conceptDriftDetector.get_scale(lower_limit_deviated_kpi, mean_kpi, num_intervals, kpi)
                map_ranges_values = conceptDriftDetector.get_scales_map(scale, expr)
                print("scale*", scale)

                for range_, val in map_ranges_values.items():
                    print(range_[0], range_[1], val)
                print("---- end ranges ----")

                tuned_w_inc = conceptDriftDetector.get_value_for_range(drift_magnitude, map_ranges_values)
                print('tuned_w_inc', tuned_w_inc)
                # tuned_w_inc = .995

                # 3. Conduct ST Surface Level Training
                # print("\t 3. In the short-term surface level retraining using new hyperparameters.")
                avg_plane_from_st_trn = surface_level_retrain_using_tuned_hyperparameters(tuned_w_inc, n1norm, n2norm,intersection_point)
                print("$$$$$$$$$$$$$$ avg_plane UPDATED THROUGH SHORT TERM LEVEL $$$$$$$$$$$$$$", avg_plane)

                inc_predicted_y_test_from_stt = Predictions.predict_(Xj, avg_plane_from_st_trn)
                acc_from_stt = Measures.accuracy(yj, inc_predicted_y_test_from_stt)
                if(acc_from_stt > current_accuracy):
                    memoryManager.remove_last_mini_batch_data()
                    add_mini_batch_statistics_to_memory(Xj, yj, avg_plane_from_st_trn, memoryManager, recomputed=True)
                    base_coeff = avg_plane_from_st_trn


            else:
                print("short term NOT detected")
                base_coeff = avg_plane

        # END CONCEPT DRIFT STUFF

        print("...updating the model...")
        print("=====================================================================================")
        print()
        print()
        print()

        # base_coeff = avg_plane
        inc_predicted_y_test = Predictions.predict_(Xj, base_coeff)
        cost = Measures.cross_entopy_cost(yj, inc_predicted_y_test)
        cost_list = np.append(cost_list, cost)
        epoch_list = np.append(epoch_list, (iteration * increment_size) + no_of_base_model_points)
        inc_acc = Measures.accuracy(yj, inc_predicted_y_test)
        acc_list = np.append(acc_list, inc_acc)

    accuracy_list = memoryManager.get_acc_list()
    cost_list_fin = memoryManager.get_cost_list()
    return base_coeff, epoch_list, cost_list_fin, accuracy_list # acc_list






def olc_wa_sccm_classification(X, y, w_inc, base_model_size, increment_size, kpi, multiplier):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    n_samples, n_features = X.shape
    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    # Step 1: Compute the base model, get its coefficients
    no_of_base_model_points = increment_size#Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    base_model_w, base_model_b = BatchClassification.logistic_regression(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions.predict(base_model_training_X, base_model_w, base_model_b)
    base_coeff = np.array(np.append(np.append(base_model_w, -1), base_model_b))
    # Plotter.plot(base_model_training_X, base_model_training_y, base_coeff)
    # base_coeff = np.array(np.append(base_model_w, base_model_b))
    cost = Measures.cross_entopy_cost(base_model_training_y, base_model_predicted_y)
    cost_list = np.append(cost_list, cost)
    epoch_list = np.append(epoch_list, no_of_base_model_points)
    base_acc = Measures.accuracy(base_model_training_y, base_model_predicted_y)
    acc_list = np.append(acc_list, base_acc)

    # Statistical Meta-Data Saved about First Mini-Batch (BaseModel)
    miniBatchMetaData = MiniBatchMetaData(None, cost, base_acc)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    num_intervals = 5  # used for the number of intervals in the scaled map.

    # INCREMENTAL_MODEL
    mini_batch_generator = get_next_mini_batch(X, y, no_of_base_model_points, increment_size)
    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:
        # reset to default
        w_inc = .5
        w_base = .5

        # 1. train using incoming mini-batch
        inc_model_w, inc_model_b = BatchClassification.logistic_regression(Xj, yj)
        inc_coeff = np.array(np.append(np.append(inc_model_w, -1), inc_model_b))
        # inc_coeff = np.array(np.append(inc_model_w, inc_model_b))

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization

        avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        # avg = (np.dot(w_base, n1) + np.dot(w_inc, n2)) / (w_base + w_inc)



        # Step 5: intersection-point = get-intersection-point(base-boundary-line, inc-boundary-line)
        # We will find an intersection point between the two models, the base and the incremental.
        # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
        # will be a weighted middle point.
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                               w_base=w_base, w_inc=w_inc)

        # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
        # In this step we define a new space as a result from the average vector 1 and the intersection point
        avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
        # Plotter.plot(Xj, yj, avg_plane)
        # CONCEPT DRIFT STUFF
        # 1. Add statistics about Mini-Batch to memory_manager
        add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=False)

        # 3. The list length min is 3
        if (len(memoryManager.mini_batch_data) >= 4):
            print("********** SHORT TERM ***********")
            # 3. Check for ST Drift
            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(memoryManager.mini_batch_data,
                                                                   kpi)  # contains last 4 elements
            print("KPI_Window_ST", KPI_Window_ST)
            threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = conceptDriftDetector.get_meaures(
                KPI_Window_ST, multiplier, kpi)

            print("threshold", threshold, "mean", mean_kpi, "prev", KPI_Window_ST[-2], "curr", KPI_Window_ST[-1],
                  "lower_limit_deviated_kpi", lower_limit_deviated_kpi, "drift_magnitude", drift_magnitude)
            ST_drift_detected = conceptDriftDetector.detect_ST_drift(KPI_Window_ST, mean_kpi, threshold, kpi)
            print("SHORT TERM DRIFT DETECTED", ST_drift_detected)
            # ST_drift_detected = conceptDriftDetector.detect(memoryManager.mini_batch_data, recomputed=False)
            if (ST_drift_detected):
                print('Short Term Drift Detected')
                # 1. remove last element in the mini_batch_data
                memoryManager.remove_last_mini_batch_data()

                scale = conceptDriftDetector.get_scale(lower_limit_deviated_kpi, mean_kpi, num_intervals, kpi)
                print("scale", scale)
                map_ranges_values = conceptDriftDetector.get_scales_map(scale)
                print("---- ranges ----")

                for range_, val in map_ranges_values.items():
                    print(range_[0], range_[1], val)
                print("---- end ranges ----")

                tuned_w_inc = conceptDriftDetector.get_value_for_range(drift_magnitude, map_ranges_values)
                print('tuned_w_inc', tuned_w_inc)
                tuned_w_inc = 1

                # 3. Conduct ST Surface Level Training
                # print("\t 3. In the short-term surface level retraining using new hyperparameters.")
                avg_plane = surface_level_retrain_using_tuned_hyperparameters(tuned_w_inc, n1norm, n2norm,
                                                                              intersection_point)
                print("$$$$$$$$$$$$$$ avg_plane UPDATED THROUGH SHORT TERM LEVEL $$$$$$$$$$$$$$", avg_plane)
                # print("\t 4. short-term memory surface level retraining finished.")
                add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=True)
            else:
                print("short term NOT detected")

        # END CONCEPT DRIFT STUFF

        print("...updating the model...")
        print("=====================================================================================")
        print()
        print()
        print()

        base_coeff = avg_plane
        inc_predicted_y_test = Predictions.predict_(Xj, base_coeff)
        cost = Measures.cross_entopy_cost(yj, inc_predicted_y_test)
        cost_list = np.append(cost_list, cost)
        epoch_list = np.append(epoch_list, (iteration * increment_size) + no_of_base_model_points)
        inc_acc = Measures.accuracy(yj, inc_predicted_y_test)
        acc_list = np.append(acc_list, inc_acc)

    accuracy_list = memoryManager.get_acc_list()
    cost_list_fin = memoryManager.get_cost_list()
    return base_coeff, epoch_list, cost_list_fin, accuracy_list # acc_list





def olc_wa_classification_KFold(X, y, w_inc, base_model_size, increment_size, seed, shuffle):
    kf = KFold(n_splits=5, random_state=seed, shuffle=shuffle)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        coeff, epoch_list, cost_list, acc_list = olc_wa_classification(X_train, y_train, w_inc, base_model_size, increment_size)

        predicted_y_test = Predictions.predict_(X_test, coeff)
        acc = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc)

    return np.array(scores).mean()

def olc_wa_sccm_classification_KFold(X, y, w_inc, base_model_size, increment_size, seed, shuffle, kpi, multiplier, expr):
    kf = KFold(n_splits=5, random_state=seed, shuffle=shuffle)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        coeff, epoch_list, cost_list, acc_list = olc_wa_lstm_sccm_classification(X_train, y_train, w_inc, base_model_size, increment_size, kpi, multiplier, expr)

        predicted_y_test = Predictions.predict_(X_test, coeff)
        acc = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc)

    return np.array(scores).mean()




